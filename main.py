import torch
from src.models import cnn_from_scratch, train_model, test_model, predict_model
from src.utils import optimizer_functions, criterion_functions
from src.data import data_loader
from glob import glob
import numpy as np
import random

use_existing = True
n_epochs = 3
model_parameters_save_name = f"model_{n_epochs}.pt"

# get all file names
dog_files = np.array(glob("data/raw/dogImages/*/*/*"))
if len(dog_files) == 0:
    raise RuntimeError("No dog images found! Make sure there is data in data/raw/dogImages")
random.seed(10)
random.shuffle(dog_files)
dog_files_sample = dog_files[:10]

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
        try:
            model.load_state_dict(torch.load(f"models/{model_parameters_save_name}"))
        except FileNotFoundError:
            raise FileNotFoundError(f"No model found at models/{model_parameters_save_name}. Train a model first or "
                                    f"put an already-trained model at that location")
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
    for dog_image in dog_files_sample:
        print(f"Predict for {dog_image}")
        predict_model.create_prediction_fig(dog_image, model)


