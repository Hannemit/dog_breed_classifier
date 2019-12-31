import torch
from src.models import cnn_from_scratch, train_model, test_model, predict_model
from src.utils import optimizer_functions, criterion_functions
from src.data import data_loader
from glob import glob
import numpy as np
import random
from src.utils import load_torch_model

use_existing = True
n_epochs = 90
model_parameters_save_name = f"model_scratch_{n_epochs}.pt"

# get all file names
dog_files = np.array(glob("data/raw/dogImages/*/*/*"))
human_files = np.array(glob("data/raw/lfw/*/*"))
if len(dog_files) == 0 or len(human_files) == 0:
    raise RuntimeError("No dog or human images found! Make sure there is data in data/raw/")
random.seed(10)
random.shuffle(dog_files)
random.shuffle(human_files)
dog_files_sample = dog_files[:10]
human_files_sample = human_files[:5]

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# define model
model = cnn_from_scratch.Net()

# select loss function
criterion = criterion_functions.cross_entropy_loss()

# select optimizer
optimizer = optimizer_functions.optimizer_adam(model.parameters(), learning_rate=0.001)

# get data loaders
data_loaders = data_loader.get_all_loaders(batch_size=15)


if __name__ == "__main__":

    # get the trained model
    if use_existing:
        # use an existing model from the specified path, we load its weights
        print(f"Using existing model at models/{model_parameters_save_name}")
        load_torch_model.load_model(model, f"models/{model_parameters_save_name}")
    else:
        # train the model. The best model is automatically saved at the specified save_path
        print(f"Training model for {n_epochs} epochs")
        model = train_model.train(n_epochs=n_epochs,
                                  loaders=data_loaders,
                                  model=model,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  use_cuda=use_cuda,
                                  save_path=f"models/{model_parameters_save_name}")
        torch.cuda.empty_cache()
        print("Training finished!")

    # test model
    print("Testing model..")
    test_model.test(model, criterion, use_cuda)

    # predict some images
    print("Predicting for some sample images...")
    class_names = data_loader.get_training_classnames()
    for dog_image in dog_files_sample:
        print(f"Predict for {dog_image}")
        predict_model.create_prediction_fig(dog_image, model, class_names)

    # also predict for some humans
    for human_image in human_files_sample:
        print(f"Predict for {human_image}")
        predict_model.create_prediction_fig(human_image, model, class_names)


