import os
import pickle


def store_pickle(filepath, object):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        object = pickle.load(f)
    return object


def safe_make_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
