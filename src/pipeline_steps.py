from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
from transformers import AutoProcessor, AutoModelForImageClassification
from torch.nn.functional import softmax
from data_loader import allen_api
import os
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from skbio.stats.composition import clr
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import binary_cross_entropy
from collections import defaultdict

load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchNeuronEmbeddingLogReg(nn.Module):
    def __init__(self, stimulus_dim, neuron_count, neuron_embed_dim=8):
        super().__init__()
        self.neuron_embed = nn.Embedding(neuron_count, neuron_embed_dim)
        self.linear = nn.Linear(stimulus_dim + neuron_embed_dim, 1)

    def forward(self, X_stim, neuron_ids):
        """
        Args:
            X_stim: (batch_size, stimulus_dim) CLR-transformed image embeddings
            neuron_ids: (batch_size,) LongTensor with neuron indices

        Returns:
            probs: (batch_size,) predicted spike probabilities
        """
        e_neuron = self.neuron_embed(neuron_ids)         # (batch_size, neuron_embed_dim)
        x = torch.cat([X_stim, e_neuron], dim=1)          # (batch_size, stimulus_dim + embed_dim)
        logits = self.linear(x).squeeze(1)                # (batch_size,)
        probs = torch.sigmoid(logits)
        return probs
    
class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass

class AllenStimuliFetchStep(PipelineStep):
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        self.boc = boc

    def process(self, data):
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {'container_id': container_id, 'session': session, 'stimulus': stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {
            'natural_movie_one': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_one'),
            'natural_movie_two': self.boc.get_ophys_experiment_data(self.SESSION_C).get_stimulus_template('natural_movie_two'),
            'natural_movie_three': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_three'),
            'natural_scenes': self.boc.get_ophys_experiment_data(self.SESSION_B).get_stimulus_template('natural_scenes')
        }

        data['raw_data_dct'] = raw_data_dct
        return data

class AllenNeuralResponseExtractor(PipelineStep):
    def __init__(self, boc, eid_dict, stimulus_session_dict, threshold=0.0):
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.threshold = threshold

    def process(self, data):
        container_id = data['container_id']
        session = data['session']
        stimulus = data['stimulus']

        valid_stims = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stims:
            raise ValueError(f"Stimulus '{stimulus}' not valid for session '{session}'. Valid: {valid_stims}")

        session_eid = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = self.boc.get_ophys_experiment_events(ophys_experiment_id=session_eid)
        stim_table = dataset.get_stimulus_table(stimulus)

        X_list, frame_list = [], []

        for _, row in stim_table.iterrows():
            if row['frame'] == -1:
                continue
            start_t, end_t = row['start'], row['end']
            frame_idx = row['frame']
            time_indices = range(start_t, end_t)

            if len(time_indices) == 0:
                trial_vector = np.zeros(dff_traces.shape[0])
            else:
                relevant_traces = dff_traces[:, time_indices]
                trial_vector = np.max(relevant_traces, axis=1)
                trial_vector = (trial_vector > self.threshold).astype(float)

            X_list.append(trial_vector)
            frame_list.append(frame_idx)

        data['X_neural'] = np.vstack(X_list)
        data['frame_ids'] = np.array(frame_list)
        return data

class ImageToEmbeddingStep(PipelineStep):
    def __init__(self, embedding_cache_dir: str):
        self.model_name = "google/vit-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.eval()

        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_prefix = self.model_name.replace('/', '_')
        self.embeddings_file = self.embedding_cache_dir / f"{self.model_prefix}_embeddings_softmax.pkl"

    def process(self, data):
        raw_data_dct = data['raw_data_dct']

        if self.embeddings_file.exists():
            print(f"Found existing embeddings for model {self.model_prefix}. Using file:\n {self.embeddings_file}")
            data['embedding_file'] = str(self.embeddings_file)
            return data

        print(f"No cache found for model {self.model_prefix}. Computing now...")
        embeddings_dict = {}
        for stim_name, frames_array in raw_data_dct.items():
            embeddings = self._process_stims(frames_array)
            embeddings_dict[stim_name] = embeddings

        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"Saved embeddings to {self.embeddings_file}")

        data['embedding_file'] = str(self.embeddings_file)
        return data

    def _process_stims(self, frames_array):
        n_frames = len(frames_array)
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)

        with torch.no_grad():
            inputs = self.processor(images=frames_3ch[0], return_tensors="pt")
            outputs = self.model(**inputs)
            n_classes = outputs.logits.shape[-1]

        all_probs = np.empty((n_frames, n_classes), dtype=np.float32)
        for i in range(n_frames):
            inputs = self.processor(images=frames_3ch[i], return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = softmax(logits, dim=-1).squeeze().cpu().numpy()
            all_probs[i, :] = probs

        return all_probs

class AllenViTRegressionDatasetBuilder(PipelineStep):
    def __init__(self):
        pass

    def process(self, data):
        embedding_file = data['embedding_file']
        stimulus = data['stimulus']
        frame_ids = data['frame_ids']

        with open(embedding_file, 'rb') as f:
            all_stim_embeddings = pickle.load(f)

        embed_array = all_stim_embeddings[stimulus]
        X_embed = np.array([embed_array[f_idx] for f_idx in frame_ids], dtype=np.float32)

        data['X_embed'] = X_embed
        return data

class StimulusGroupKFoldSplitterStep(PipelineStep):
    def __init__(self, boc, eid_dict, stimulus_session_dict, n_splits=10):
        """
        :param boc: Allen BrainObservatoryCache
        :param eid_dict: container_id -> { session: eid }
        :param stimulus_session_dict: e.g. {'three_session_A': [...], ...}
        :param n_splits: how many CV folds
        """
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.n_splits = n_splits

    def process(self, data):
        """
        data requires 'container_id', 'session', 'stimulus'.
        Creates data['folds'] => list of (X_train, frames_train, X_test, frames_test).
        """
        container_id = data['container_id']
        session = data['session']
        stimulus = data['stimulus']
        
        valid_stims = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stims:
            raise ValueError(f"Stimulus '{stimulus}' not valid for session '{session}'. "
                             f"Valid: {valid_stims}")

        session_eid = self.eid_dict[container_id][session]

        dataset = self.boc.get_ophys_experiment_data(session_eid)
        
        #dff_traces = dataset.get_dff_traces()[1]  # shape (n_neurons, n_timepoints)
        dff_traces= self.boc.get_ophys_experiment_events(ophys_experiment_id=session_eid)
        #dff_traces = dataset

        stim_table = dataset.get_stimulus_table(stimulus)
        print(stim_table)


        X_list, frame_list, groups = [], [], []

        for _, row_ in stim_table.iterrows():
            if row_['frame']!=-1:
                start_t, end_t = row_['start'], row_['end']
                frame_idx = row_['frame']
                time_indices = range(start_t, end_t)

                if len(time_indices) == 0:
                    trial_vector = np.zeros(dff_traces.shape[0])
                else:
                    relevant_traces = dff_traces[:, time_indices]
                    #trial_vector = np.max(relevant_traces, axis=1)
                    threshold = 0.0  # or pick something domain-appropriate
                    trial_vector = np.max(relevant_traces, axis=1)

                    # Convert to binary: 1 if above threshold, else 0
                    trial_vector = (trial_vector > threshold).astype(float)
                
                X_list.append(trial_vector)
                frame_list.append(frame_idx)
                groups.append(frame_idx)
            else:
                pass

        X = np.vstack(X_list)
        print(X.shape)
        frames = np.array(frame_list)
        groups = np.array(groups)

        folds = []
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            frames_train, frames_test = frames[train_idx], frames[test_idx]
            folds.append((X_train, frames_train, X_test, frames_test))

        data['folds'] = folds
        return data


class MergeEmbeddingsStep(PipelineStep):
    """
    Reads the embedding file from data['embedding_file'],
    merges it with each fold in data['folds'], resulting in data['merged_folds'].
    """

    def __init__(self):
        # If you prefer, you can pass an argument here, e.g. `embedding_file`, 
        # but in this design, we read it from data.
        pass

    def process(self, data):
        """
        We expect:
          data['embedding_file'] -> path to a pickle file containing a dict: {stim_name: 2D array of embeddings}
          data['folds'] -> list of (X_train, frames_train, X_test, frames_test)
          data['stimulus'] -> e.g. 'natural_movie_one'
        
        We'll create data['merged_folds'] = list of (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test).
        """
        embedding_file = data['embedding_file']
        stimulus = data['stimulus']
        folds = data['folds']

        # Load embeddings
        with open(embedding_file, 'rb') as f:
            all_stim_embeddings = pickle.load(f)

        # e.g. shape (#frames_in_stim, embedding_dim)
        # Note: we assume the indexing in all_stim_embeddings[stimulus]
        # matches the 'frame_idx' from the Allen table.
        embed_array = all_stim_embeddings[stimulus]
        #print(embed_array.shape)

        merged_folds = []
        for (Xn_train, frames_train, Xn_test, frames_test) in folds:
            # Build Xe_train from embed_array
            Xe_train = np.array([embed_array[f_idx] for f_idx in frames_train], dtype=np.float32)
            Xe_test  = np.array([embed_array[f_idx] for f_idx in frames_test], dtype=np.float32)

            merged_folds.append((Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test))

        data['merged_folds'] = merged_folds
        return data


class StimNeuronDataset(Dataset):
    def __init__(self, X_embed_clr, X_neural, neuron_ids):
        """
        X_embed_clr: (n_timepoints, embed_dim), already CLR-transformed
        X_neural: (n_timepoints, n_neurons), binary labels
        neuron_ids: list of neuron indices
        """
        self.samples = []
        for neuron_id in neuron_ids:
            for i in range(X_embed_clr.shape[0]):
                self.samples.append((X_embed_clr[i], neuron_id, X_neural[i, neuron_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, nid, y = self.samples[idx]
        return x.astype(np.float32), nid, y

class MultiTaskLogRegModel(nn.Module):
    def __init__(self, x_dim, n_neurons, neuron_embed_dim=16):
        super().__init__()
        self.neuron_embed = nn.Embedding(n_neurons, neuron_embed_dim)
        self.linear = nn.Linear(x_dim + neuron_embed_dim, 1)

    def forward(self, x, neuron_ids):
        nvec = self.neuron_embed(neuron_ids)
        x_all = torch.cat([x, nvec], dim=1)
        return torch.sigmoid(self.linear(x_all)).squeeze(-1)

class MultiTaskLogRegStep:
    def __init__(self, C=1.0, lr=1e-2, n_epochs=10, batch_size=256, alpha=0.05, n_jobs=1):
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_jobs = n_jobs

    def process(self, data):
        merged_folds = data['merged_folds']
        n_neurons = merged_folds[0][0].shape[1]
        embed_dim = merged_folds[0][1].shape[1]
        rng = np.random.default_rng(seed=42)

        def train_and_eval(fold_data):
            Xn_train, Xe_train_raw, Xn_test, Xe_test_raw, _, _ = fold_data

            # CLR-transform
            Xe_train = np.apply_along_axis(lambda x: clr(x + 1e-6), 1, Xe_train_raw).astype(np.float32)
            Xe_test = np.apply_along_axis(lambda x: clr(x + 1e-6), 1, Xe_test_raw).astype(np.float32)

            #Xe_train_perm = Xe_train[rng.permutation(Xe_train.shape[0])]
            Xe_train_perm = np.array([vec[rng.permutation(vec.shape[0])] for vec in Xe_train])

            # Datasets
            train_ds_real = StimNeuronDataset(Xe_train, Xn_train, list(range(n_neurons)))
            train_ds_perm = StimNeuronDataset(Xe_train_perm, Xn_train, list(range(n_neurons)))
            test_ds = StimNeuronDataset(Xe_test, Xn_test, list(range(n_neurons)))

            train_loader_real = DataLoader(train_ds_real, batch_size=self.batch_size, shuffle=True)
            train_loader_perm = DataLoader(train_ds_perm, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            def train_model(train_loader):
                model = MultiTaskLogRegModel(embed_dim, n_neurons)
                opt = torch.optim.Adam(model.parameters(), lr=self.lr)
                model.train()
                for _ in range(self.n_epochs):
                    print(_,self.n_epochs)
                    for xb, nid, yb in train_loader:
                        xb, nid, yb = xb, nid.long(), yb.float()
                        preds = model(xb, nid)
                        weights = torch.where(yb == 1, torch.tensor(100.0), torch.tensor(1.0))
                        loss = F.binary_cross_entropy(preds, yb, weight=weights)
                        #loss=F.binary_cross_entropy(preds, yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                return model

            model_real = train_model(train_loader_real)
            model_perm = train_model(train_loader_perm)

            model_real.eval()
            model_perm.eval()

            y_true_dict = defaultdict(list)
            y_pred_real_dict = defaultdict(list)
            y_pred_perm_dict = defaultdict(list)

            with torch.no_grad():
                for xb, nid, yb in test_loader:
                    preds_real = model_real(xb, nid.long()).numpy()
                    preds_perm = model_perm(xb, nid.long()).numpy()
                    for i in range(len(xb)):
                        n = int(nid[i])
                        if yb[i] == 1:
                            y_true_dict[n].append(1)
                            y_pred_real_dict[n].append(preds_real[i])
                            y_pred_perm_dict[n].append(preds_perm[i])

            return y_pred_real_dict, y_pred_perm_dict

        # Parallel across folds
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(train_and_eval)(fold_data) for fold_data in merged_folds
        )

        # Aggregate
        pred_real_all = defaultdict(list)
        pred_perm_all = defaultdict(list)

        for real_dict, perm_dict in results:
            for nid in real_dict:
                pred_real_all[nid].extend(real_dict[nid])
            for nid in perm_dict:
                pred_perm_all[nid].extend(perm_dict[nid])

        n_neurons = max(max(pred_real_all.keys(), default=-1), max(pred_perm_all.keys(), default=-1)) + 1
        nll_real = np.full(n_neurons, np.nan)
        nll_perm = np.full(n_neurons, np.nan)

        for nid in range(n_neurons):
            try:
                if pred_real_all[nid]:
                    nll_real[nid] = log_loss(np.ones_like(pred_real_all[nid]), pred_real_all[nid], labels=[0, 1])
                if pred_perm_all[nid]:
                    nll_perm[nid] = log_loss(np.ones_like(pred_perm_all[nid]), pred_perm_all[nid], labels=[0, 1])
            except Exception:
                continue

        data['multi_task_event_nll_real'] = nll_real
        data['multi_task_event_nll_perm'] = nll_perm
        data['multi_task_event_nll_delta'] = nll_real - nll_perm

        with open("multi_task_nll_event_only_random.pkl", "wb") as f:
            pickle.dump({
                'real': nll_real,
                'perm': nll_perm,
                'delta': nll_real - nll_perm
            }, f)

        print("Saved multitask NLL (event-only) to multi_task_nll_event_only.pkl")
        return data

class AnalysisPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data

def make_container_dict(boc):
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        c_id, sess_type, ids = row
        if c_id not in eid_dict:
            eid_dict[c_id] = {}
        eid_dict[c_id][sess_type] = ids[0]
    return eid_dict
    
if __name__ == '__main__':
    boc = allen_api.get_boc()
    eid_dict = make_container_dict(boc)
    stimulus_session_dict = {
        'three_session_A': ['natural_movie_one', 'natural_movie_three'],
        'three_session_B': ['natural_movie_one', 'natural_scenes'],
        'three_session_C': ['natural_movie_one', 'natural_movie_two'],
        'three_session_C2': ['natural_movie_one', 'natural_movie_two']
    }

    embedding_cache_dir = os.environ.get('TRANSF_EMBEDDING_PATH', 'embeddings_cache')
    container_id = list(eid_dict.keys())[0]
    session = list(eid_dict[container_id].keys())[0]
    stimulus = stimulus_session_dict.get(session, [])[0]
    session='three_session_B'
    stimulus='natural_scenes'
    print(f"Running pipeline for container_id={container_id}, session={session}, stimulus={stimulus}")

    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(embedding_cache_dir),
        StimulusGroupKFoldSplitterStep(boc, eid_dict, stimulus_session_dict),
        MergeEmbeddingsStep(),
        MultiTaskLogRegStep()
    ])
    import time
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))
    end=time.time()
    print('Time taken', end-start)