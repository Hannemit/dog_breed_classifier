from torchvision import datasets
import torch
from src.data import data_transformer
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_DATA_PATH = "data/raw/dogImages/train"
VALIDATION_DATA_PATH = "data/raw/dogImages/valid"
TEST_DATA_PATH = "data/raw/dogImages/test"


def get_loader(path_to_images, transformer, batch_size, num_workers, shuffle=True):
    data = datasets.ImageFolder(path_to_images, transform=transformer)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def get_all_loaders(batch_size=15):
    train_transformer = data_transformer.data_transform_from_scratch
    test_valid_transformer = data_transformer.data_transform_bare

    train_loader = get_loader(TRAIN_DATA_PATH, train_transformer, batch_size=batch_size, num_workers=0)
    valid_loader = get_loader(VALIDATION_DATA_PATH, test_valid_transformer, batch_size=batch_size, num_workers=0)
    test_loader = get_loader(TEST_DATA_PATH, test_valid_transformer, batch_size=batch_size, num_workers=0)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


def get_test_loader(batch_size=15, path=""):
    if not path:
        path = TEST_DATA_PATH
    return get_loader(path, data_transformer.data_transform_bare, batch_size=batch_size, num_workers=0)


def get_training_classnames(data_path=""):
    """
    Get the names of all the classes that are present in the training data
    :return: array of strings, each string i sa class name (a breed of dog)
    """
    if not data_path:
        data_path = TRAIN_DATA_PATH
    data = datasets.ImageFolder(data_path, transform=data_transformer.data_transform_from_scratch)
    class_names = np.array([item[4:].replace("_", " ") for item in data.classes])
    return class_names
