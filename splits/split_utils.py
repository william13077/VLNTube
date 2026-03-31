import json

_trainval_cache = None

def is_trainval(splits_file, scene_name):
    """Check if scene_name is in the trainval split.
    Returns True if it should be processed, False if it should be skipped.
    Caches the trainval set after first load.
    """
    global _trainval_cache
    if _trainval_cache is None:
        with open(splits_file) as f:
            _trainval_cache = set(json.load(f)['trainval'])
    if scene_name not in _trainval_cache:
        print(f'Skipping scene {scene_name}: not in trainval split')
        return False
    return True
