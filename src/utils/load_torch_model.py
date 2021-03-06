import torch
from functools import partial
import pickle

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")


def load_model(model, model_path):
    """
    Load a pytorch model that was saved earlier.
    :param model: the model that we want to load the state dict for
    :param model_path: string, path to the model
    :return:
    """
    try:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage, pickle_module=pickle))
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found at {model_path}. Train a model first or "
                                f"put an already-trained model there")