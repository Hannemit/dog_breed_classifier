dog breed classifier
==============================

In this project we use a CNN to predict dog breeds. In total, we have 133 classes (the breeds) and a new images is classified as one of these. Specifically, we:

1. train a simple CNN (just a handful of convolutional layers and a few dense layers)
2. define a human face detector to detect whether a human face is in an image (we use a pre-trained face detector)
3. define a dog detector, detecting whether a dog is present in an image (using a pre-trained VGG16 model)

Given a completely new image, we then
* find out whether the image is of a human
* find out whether the image is of a dog
* Depending on the outcome of the above steps, we do the following:  
    * no dog and no human --> throw an error, a better image needs to be supplied
    * human and no dog    --> find the closest dog breed to this human using our trained CNN. Output a nice figure with human and closest dog breed
    * dog and no human --> predict the dog breed using our trained CNN. Output a nice figure showing the closest dog breed
    * human and dog --> do same for if human and print out some extra messages
    
  

Additionally, we will first detect whether a human is present in the image (if so, we'll output a slightly different figure which shows the closest dog breeds to that human). 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## How to use

* Make sure the correct environment is set up (see requirements)
* Make sure the data is downloaded and in the correct place
* run `main.py` by calling `python main.py` from the root of the project. Make sure to set some of the parameters at the start of the file e.g. the number of epochs to run for and whether to train or use an existing model

when running `main.py`, we 
1. either train a model or load a previously trained model
2. print out the accuracy and loss of this model on test data
3. predict breeds of some random images using the model, the predict outputs are saved in `src/visualization`

When wanting to test a particular model, we can also directly call `python src/models/test_model models/my_model` where the argument should be the path to a trained model. This will print out the accuracy and loss for this model on the test data. 

### Note on training
When not using a GPU, training will be slow. As an example of how the code works, first set `use_existing = False` and `n_epochs = 3` in `main.py` to train a model for 3 epochs (this is also doable without GPU). Running `main.py` again with `use_existing = True` should now use this newly trained model to make predictions. Of course these predictions will be very bad, but it illustrates the flow of the code and outputs some nice images in `src/visualization`.

## Requirements


##### Create virtual environment
Create a virtual environment using (linux)  
`conda create -n dog_breed python=3.7.5`
 
 or, for windows  
 `conda create -n dog_breed python=3.7.5 numpy==1.17.3`

#### install other requirements
`cd` into `dog_breed_classifier` (i.e. go to the root of the project), activate the virtual environment with 

`conda activate dog_breed`

then, install the requirements  

`pip install -r requirements.txt`

If an error for installing torch occurs, see trouble shooting.

## Notebooks
In the notebooks, the simple CNN model is trained using a GPU for about 90 epochs and a test accuracy of almost 50% is reached. This is not bad given that we have 133 different classes, and the CNN model is really quite simple. 

Another notebook uses transfer learning and replaces the last dense layer of the VGG16 model to finetune it on our dog breed dataset.  Weights of earlier layers are frozen. This leads to a much higher accuracy (since the model is more complicated and has been trained on ImageNet). 

## The data
#### dog data

* The dog images are available at 
`https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip` 
* Once downloaded, unzip and put the `dogImages` folder in `data/raw`. There will automatically be train, test and valid folders inside the dogImages folder.

#### Human data
* Human images are available at 
`https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip`
* Once downloaded, unzip and put the folder into `data/raw`

## Troubleshooting

* If there are problems installing `torch` (as there were for me initially), try this:  
`pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl`  
`pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl`  
Or, if that still doesn't work, try  
`pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html`
and then re-run `pip install -r requirements.txt`
* If an error is thrown similar to `error: (-215:Assertion failed) !empty() in function 'cv::CascadeClassifier::detectMultiScale'`, then find the path to the `haarcascade_frontalface_default.xml` file, and insert that in `src/models/detect_faces.py` at the top, instead of the path that's currently there.
