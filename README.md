Dog breed classifier
==============================

In this project, we will try to predict dog breeds. Given an image of a dog, what breed of dog is it? 

This kind of problem is perfect for Convolutional Neural Networks (CNNs). We will therefore construct a CNN (two, actually!) and we expect that these models will perform a decent job at classifying dogs. They will not be perfect, but we should definitely be able to do a lot better than random chance. 

Our data is as follows:
* 8351 different dog images
* consisting of 133 different breeds

We want to train a CNN to take an image as an input, and predict a probability for each of the 133 different classes (breeds). We can then also feed the model an image of a human, to find the dogs he/she/other_pronoun resembles most!

Specifically, we:

1. train a simple CNN (just a handful of convolutional layers and a few dense layers) which we build from scratch using PyTorch
2. define a human face detector to detect whether a human face is in an image (we use a pre-trained face detector)
3. define a dog detector, detecting whether a dog is present in an image (using a pre-trained VGG16 model)

Given a completely new image, we then
* find out whether the image is of a human
* find out whether the image is of a dog
* find the closest dog breed to the input image and output a nice figure showing the top 5 most likely breeds. The title of the image will be different depending on whether a dog, a human, both or neither were detected.
    
We then create another CNN, using a pre-trained VGG16 model. Using transfer learning, we are able to create a model that performs quite well. We remove the last dense layer of VGG16 (as it was originally trained to predict for 1000 classes on ImageNet data) and replace it with a dense layer with 133 outputs. We freeze all the weights and only fine-tune the last layer using our data. 

Results
---------------
* Using our CNN from scratch, we reach 47% accuracy. This is pretty good, considering the relatively simple architecture (see below) and the fact that random chance (ignoring class imbalances) would have given us only 1% accuracy
* Using our fine-tuned VGG16 model, we reach about 85% accuracy

The performance of our models during training is measured by using the cross entropy loss, a standard loss function for classification problems. 

Below is an example of a correct prediction! The input image is shown (left), together with a bar plot of the top 5 most likely dog breeds, as well as an example of the overall most likely breed (right). This visualization is created in `src/models/predict_model.py`, more examples are shown in the notebooks.

![](./src/visualization/prediction_Affenpinscher_00069.jpg?raw=true "")



Model architecture choices
--------------
For the CNN model we build ourselves, the final architecture is shown in the figure below. There are five convolutional layers in total, after each convolution we i) apply a ReLu activation function, ii) perform a max-pooling to downsize the image by a factor of 2 in both the x and y dimension, iii) perform batch normalization. Dropout is used after the first two fully connected layers. 

![](./src/visualization/cnn_architecture.png?raw=true "")


A number of different architectures were tried out, such as:
* 3, 4 or 5 convolutional layers
* batch normalization or not after the convolutional layers
* a learning rate of 0.01/0.005/0.001/0.0005 for the Adam optimizer.
* various output sizes of the fully connected layers, to get towards the final size of 133

The final architecture used is the one that seemed to work best out of the ones that were tried. 




Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                          e.g.`1_initial_data_exploration`.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data. In this project we just use it to create our data loaders
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations, we also save figures here
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## How to use

* make sure the requirements in requirements.txt are installed (see requirements below) and the virtual environment is activated
* run `python -m ipykernel install --user --name dog_breed --display-name "Python (dog_breed)"` to be able to easily use the virtual environment in a jupyter notebook
* Download the data and put the unzipped folders in `data/raw` (see the data section below)
* run `python main.py` from the root of the project. Make sure to set some of the parameters at the start of the file e.g. the number of epochs to run for and whether to train or use an existing model

when running `main.py`, we 
1. either train a model or load a previously trained model
2. print out the accuracy and loss of this model on test data
3. predict breeds of some random images using the model, the predicted outputs are saved in `src/visualization`

When wanting to test a particular model, we can also directly call `python src/models/test_model models/my_model` where the argument (i.e. `models/my_model`) should be the path to a trained model. This will print out the accuracy and loss for this model on the test data. 

### Note on training
When not using a GPU, training will be slow. As an example of how the code works, first set `use_existing = False` and `n_epochs = 3` in `main.py` to train a model for 3 epochs (this is also doable without GPU). Running `main.py` again with `use_existing = True` should now use this newly trained model to make predictions. Of course these predictions will be very bad, but it illustrates the flow of the code and outputs some nice images in `src/visualization`.

## Requirements


##### Create virtual environment
Create a virtual environment using
`conda create -n dog_breed python=3.7.5`

#### install other requirements
`cd` into `dog_breed_classifier` (i.e. go to the root of the project), activate the virtual environment with 

`conda activate dog_breed`

then, install the requirements  

`pip install -r requirements.txt`

If an error for installing torch occurs, follow the steps in the trouble shooting section below. NOTE: make sure the path to the `haarcascade_frontalface_alt.xml` file in `src/models/detect_faces.py` is correct, see the Troubleshooting section for more details on this. 

## Data
#### dog data

* The dog images are available at 
`https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip` 
* Once downloaded, unzip and put the `dogImages` folder in `data/raw`. There will automatically be train, test and valid folders inside the dogImages folder.

#### Human data
* Human images are available at 
`https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip`
* Once downloaded, unzip and put the `lfw` folder into `data/raw` (note, if there's another `lfw` folder in the `lfw` folder, then only put the inner-most in `data/raw`.

No real pre-processing is required for this data. The only processing we do is when we define our data transformers in pytorch, which will transform each input image before it goes into the CNN. These transformations include things like resizing the image, rotating it, scaling it, etc.. and they are used to augment the dataset and to prevent overfitting.

## Notebooks

There are three notebooks available in the `notebooks/` directory. The first notebook is slightly messy, and much of it is repeated in the other two notebooks (the other two notebooks are a bit neater and use code that is rewritten and put into python files, rather than defined within the notebook). 
* `1_dog_app_pytorch` creates a CNN from scratch to classify dogs. It goes through most of the code in the rest of the project step by step, and all the functions are defined within the notebook (this was before they were moved into python files). This notebook trained our CNN using a GPU for about 90 epochs and a test accuracy of almost 50% is reached. This is not bad given that we have 133 different classes, and the CNN model is really quite simple. 
* `2_cnn_from_scratch` performs similar steps as `1_dog_app_pytorch` but now uses the functions within the rest of the project. We create a CNN from scratch
* `3_vgg16_transfer` uses a pre-trained VGG16 model to classify the dogs, we replace the last dense layer by a new dense layer with 133 outputs (the number of classes we have). Weights of earlier layers are frozen. It reaches a much higher accuracy than our CNN from scratch (which is not surprising since the model is more complicated and has been trained on ImageNet). . 

To run the notebooks, just run `jupyter notebook` from within the virtual environment and this will open a web browser. The notebooks in `/notebooks/` can now be opened and run. Make sure to pick the correct virtual environment by changing the kernel (go to `Kernel` --> `Change kernel` and pick the dog_breed kernel) which makes sure we can import from `src`. 

## Conclusion
Classifying dog breeds is hard, I probably can't get 100% accuracy myself! Nevertheless, we have shown that a simple CNN can perform a very decent job. Sure, using a pre-trained model still works better, but I was surprised about how "easy" it was to write a model myself which clearly learns a lot. 

It doesn't take long to write the CNN in PyTorch at all, most of the time will be dedicated to tweaking parameters and experimenting with the number of layers (and of course waiting for the training results). I found out that:
* increasing the number of filters too quickly is not always a good idea
* a too-large batch size can also lead to worse results
* Increasing the number of convolutional layers increased my accuracy, though the returns are diminishing.

## Troubleshooting

* If there are problems installing `torch` (as there were for me initially), try this:  
`pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl`  
`pip3 install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp36-cp36m-win_amd64.whl`  
Or, if that still doesn't work, try  
`pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html`
and then re-run `pip install -r requirements.txt`
* If an error is thrown similar to `error: (-215:Assertion failed) !empty() in function 'cv::CascadeClassifier::detectMultiScale'`, then find the path to the `haarcascade_frontalface_default.xml` file, and insert that in `src/models/detect_faces.py` at the top, instead of the path that's currently there.
* If `src` is not recognized in the notebooks, make sure to set the virtual environment as the kernel (see the Notebooks section above). 
