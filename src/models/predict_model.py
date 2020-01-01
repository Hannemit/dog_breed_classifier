from src.models import detect_faces, detect_dogs, cnn_from_scratch
from src.data.data_transformer import data_transform_bare
from src.data import data_loader
from src.utils import load_torch_model
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
import torch


use_cuda = torch.cuda.is_available()
dog_detector_model = detect_dogs.get_dog_detector_model()


def get_top_predictions(img_path: str, model, class_names):
    """
    Given an image, a model, and the names of all of the classes, apply the model and return
    the prediction with highest probability, together with its class name
    :param img_path: string, path to image
    :param model: trained model
    :param class_names: list of strings, names of the classes
    :return: (float, string), the top probability and the top class name
    """
    model.eval()

    image = Image.open(img_path).convert('RGB')
    image = data_transform_bare(image)
    image.unsqueeze_(0)

    if use_cuda:
        image = image.to('cuda')
    prediction = model(image)
    probs = F.softmax(prediction, dim=1)
    top_probs, top_idx = probs.topk(5)
    top_probs = top_probs.cpu().detach().numpy()[0]

    return top_probs, class_names[top_idx.cpu().numpy()[0]]


def create_prediction_fig(img_path: str, model_predict, class_names, save_image=True, show_image=False, path_dog_files=""):
    """
    Create figure from prediction
    :param img_path: string, path to image we want to predict for
    :param model_predict: the trained model that does the predicting
    :param class_names: list of strings, all the class names (need that to output class in the figure)
    :param save_image: boolean, if True we save the image
    :param show_image: boolean, if True we call fig.show() (for use in notebook)
    :return:
    """
    # TODO: move to visualization folder
    # get all file names and class names
    if not path_dog_files:
        path_dog_files = "data/raw/dogImages/*/*/*"
    dog_files = np.array(glob(path_dog_files))
    if len(dog_files) == 0:
        raise RuntimeError("No dog files found!")

    # initialise, set model in evaluation stage (no dropout used, etc..)
    model_predict.eval()
    is_human = is_dog = False
    use_sup_title = False

    # check presence of human or dog
    if detect_faces.face_detector(img_path):
        is_human = True

    if detect_dogs.dog_detector(img_path, dog_detector_model):
        is_dog = True

    if not is_human and not is_dog:
        raise Exception(f"No human or dog detected in {img_path}... supply an image with a human or a dog")

    image = Image.open(img_path)
    probs, dog_names = get_top_predictions(img_path, model_predict, class_names)

    if is_human and is_dog:
        title = 'Both human and dog detected... This human/dog is predicted to be a .... {}!'.format(dog_names[0])
        use_sup_title = True
    elif is_human:
        title = 'This human looks like a ...{}!'.format(dog_names[0])
    else:
        title = 'This dog is predicted to be a .... {}!'.format(dog_names[0])

    top_dog = dog_names[0].replace(' ', '_')
    dog_image_idx = np.where([top_dog in name for name in dog_files])[0][0]
    image_dog = Image.open(dog_files[dog_image_idx])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.ravel()
    axs[2].imshow(image_dog)
    axs[2].axis('off')
    axs[2].set_title(f"Example of {dog_names[0]}")

    xs = np.arange(1, len(probs) + 1)
    axs[1].set_xticks(xs)
    axs[1].bar(xs, probs, align='center', alpha=0.5)
    axs[1].set_xticklabels(dog_names, rotation='vertical', fontsize=10)
    axs[1].set_title("Top dog predictions")

    axs[0].imshow(image)
    if use_sup_title:
        fig.suptitle(title + "\n")
        axs[0].set_title("Input image with both dog and human!")
    else:
        axs[0].set_title(title)
    axs[0].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_image:
        image_name = img_path.split("\\")[-1]
        image_name = image_name.split("/")[-1]
        assert image_name[-4:] == ".jpg" or image_name[-4:] == ".png", \
            f"Image {img_path} split by \\ should end in something.jpg or .png"
        plt.savefig(f"src/visualization/prediction_{image_name}")
    if show_image:
        plt.show()
    plt.close()


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        raise ValueError("Supply two arguments, 1) the path to the image we want to predict, "
                         "2) the path to a model you want to predict with, e.g. call "
                         "python src/models/predict_model.py data/raw/dogImages/test/001.Affenpinscher/Affenpinscher_00023.jpg models/my_model.pt")
    model_path = args[1]
    image_path = args[0]

    model = cnn_from_scratch.Net()
    load_torch_model.load_model(model, model_path)

    print(f"Predicting with model {model_path} for image {image_path}. Figure will be saved to src/visualization")
    class_names = data_loader.get_training_classnames()
    create_prediction_fig(image_path, model, class_names=class_names)


if __name__ == "__main__":
    main()
