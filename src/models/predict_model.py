from src.models import detect_faces, detect_dogs
from src.data.data_transformer import data_transform_bare
from src.data import data_loader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from glob import glob
import torch

# get all file names and class names
dog_files = np.array(glob("/data/dog_images/*/*/*"))
use_cuda = torch.cuda.is_available()
dog_detector_model = detect_dogs.get_dog_detector_model()


def get_top_predictions(img_path: str, model, class_names):
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


def create_prediction_fig(img_path: str, model_predict, class_names, save_image=True, show_image=False):

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
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.ravel()
    elif is_human:
        title = 'This human looks like a .....{}!'.format(dog_names[0])
        top_dog = dog_names[0].replace(' ', '_')
        print(top_dog)
        print(np.where([top_dog in name for name in dog_files]))
        dog_image_idx = np.where([top_dog in name for name in dog_files])[0][0]
        image_dog = Image.open(dog_files[dog_image_idx])
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs = axs.ravel()
        axs[2].imshow(image_dog)
        axs[2].axis('off')
    else:
        title = 'This dog is predicted to be a .... {}!'.format(dog_names[0])
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = axs.ravel()

    xs = np.arange(1, len(probs) + 1)
    axs[1].set_xticks(xs)
    axs[1].bar(xs, probs, align='center', alpha=0.5)
    axs[1].set_xticklabels(dog_names, rotation='vertical', fontsize=10)
    axs[1].set_title("Top dog predictions")

    axs[0].imshow(image)
    if use_sup_title:
        fig.suptitle(title)
    else:
        axs[0].set_title(title)
    axs[0].axis('off')

    plt.tight_layout()

    if save_image:
        image_name = img_path.split("\\")[-1]
        assert image_name[-4:] == ".jpg", f"Image {img_path} split by \\ should end in something.jpg"
        plt.savefig(f"src/visualization/prediction_{image_name}")
    if show_image:
        plt.show()
    plt.close()
