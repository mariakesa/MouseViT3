from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# This class uses a 'manifest' to keep track of downloaded data and metadata.  
# All downloaded files will be stored relative to the directory holding the manifest
# file.  If 'manifest_file' is a relative path (as it is below), it will be 
# saved relative to your working directory.  It can also be an absolute path.
allen_cache_path = os.environ.get('CAIM_ALLEN_CACHE_PATH')
boc =  BrainObservatoryCache(
    manifest_file=str(Path(allen_cache_path) / 'brain_observatory_manifest.json'))

# Download a list of all targeted areas
targeted_structures = boc.get_all_targeted_structures()
print("all targeted structures: " + str(targeted_structures))