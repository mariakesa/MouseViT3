import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from collections import defaultdict
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
allen_cache_path = Path(os.environ.get('CAIM_ALLEN_CACHE_PATH'))
assert allen_cache_path.exists(), "CAIM_ALLEN_CACHE_PATH must exist and be set in .env"

stimulus_session_dict = {
    'three_session_A': ['natural_movie_one', 'natural_movie_three'],
    'three_session_B': ['natural_movie_one', 'natural_scenes'],
    'three_session_C': ['natural_movie_one', 'natural_movie_two'],
    'three_session_C2': ['natural_movie_one', 'natural_movie_two']
}

stimulus = 'natural_scenes'
target_num_trials = 50
target_num_frames = 118
total_trials = target_num_trials * target_num_frames  # = 5900

boc = BrainObservatoryCache(
    manifest_file=str(allen_cache_path / 'brain_observatory_manifest.json'))

# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def make_container_dict(boc):
    df = pd.DataFrame(boc.get_ophys_experiments())
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped.itertuples(index=False):
        cid, stype, ids = row
        if cid not in eid_dict:
            eid_dict[cid] = {}
        eid_dict[cid][stype] = ids[0]
    return eid_dict

def get_valid_session_ids(boc, stimulus='natural_scenes'):
    eid_dict = make_container_dict(boc)
    valid_sessions = []
    for container_id, sessions in eid_dict.items():
        for session_type, eid in sessions.items():
            if session_type in stimulus_session_dict:
                if stimulus in stimulus_session_dict[session_type]:
                    valid_sessions.append(eid)
    return valid_sessions

def get_sorted_dff_trials(dff_traces, stim_table, threshold=0.0):
    """
    Return a dictionary: frame_idx -> list of response vectors, sorted by stimulus onset.
    """
    frame_trials = defaultdict(list)

    for _, row in stim_table.iterrows():
        frame_idx = row['frame']
        if frame_idx == -1:
            continue

        start, end = row['start'], row['end']
        time_indices = range(start, end)

        if len(time_indices) == 0:
            trial_vector = np.zeros(dff_traces.shape[0])
        else:
            trace_window = dff_traces[:, time_indices]
            trial_vector = np.max(trace_window, axis=1)
            trial_vector = (trial_vector > threshold).astype(float)

        frame_trials[frame_idx].append((start, trial_vector))

    # Sort trials within each frame by onset time
    sorted_trials = {
        frame_idx: [trial for _, trial in sorted(trials, key=lambda x: x[0])]
        for frame_idx, trials in frame_trials.items()
    }

    return sorted_trials

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

session_ids = get_valid_session_ids(boc, stimulus)
neural_responses = []

print(f"Processing {len(session_ids)} sessions...")

for sid in tqdm(session_ids):
    try:
        dataset = boc.get_ophys_experiment_data(sid)
        dff_traces = boc.get_ophys_experiment_events(sid)
        stim_table = dataset.get_stimulus_table(stimulus)
        stim_table = stim_table[stim_table['frame'] != -1]  # exclude blank

        frame_trials = get_sorted_dff_trials(dff_traces, stim_table)

        num_neurons = dff_traces.shape[0]
        response_matrix = np.full((num_neurons, total_trials), np.nan)

        for frame_idx in range(target_num_frames):
            trials = frame_trials.get(frame_idx, [])
            trials = trials[:target_num_trials]  # if more than 50, trim
            for trial_num, trial_vector in enumerate(trials):
                col_idx = frame_idx * target_num_trials + trial_num
                response_matrix[:, col_idx] = trial_vector

        neural_responses.append((sid, response_matrix))

    except Exception as e:
        print(f"❌ Failed to process session {sid}: {e}")
        continue

# ──────────────────────────────────────────────────────────────────────────────
# SAVE MATRICES
# ──────────────────────────────────────────────────────────────────────────────

all_data = {
    sid: response for sid, response in neural_responses
    if response.shape[1] == total_trials
}
print(f"\n✅ Collected {len(all_data)} complete aligned session matrices.")

save_path = Path("neural_activity_matrices")
save_path.mkdir(exist_ok=True)

for sid, matrix in all_data.items():
    np.save(save_path / f"{sid}_neural_responses.npy", matrix)
