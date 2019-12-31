import torchvision.transforms as transforms
from PIL import ImageFile, Image
import torch
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True
use_cuda = torch.cuda.is_available()


def vgg16_predict(img_path: str, model):
    """
    Use pre-trained VGG-16 model to obtain index corresponding to predicted ImageNet class for image at specified path
    :param img_path: string, path to image
    :param model: the trained vgg16 model, obtained from torchvision.models
    :return: int, index corresponding to VGG-16 model's prediction
    """
    # select the image
    im = Image.open(img_path).convert('RGB')

    # crop images to correct size (224x224 for vgg16), transform to tensor and normalize
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = data_transform(im)

    # include batch size dimension
    image.unsqueeze_(0)
    if use_cuda:
        image = image.to('cuda')

    # get prediction and select most likely class
    outcome = model.forward(image)
    max_index = torch.max(outcome, 1)[1].item()

    return max_index  # predicted class index


def dog_detector(img_path: str, model):
    """
    Predict whether a dog is present in the image provided.
    :param img_path: string, path to where image is
    :param model: vgg16 model, assumed to have dog predictions for indices [151, 268] (pretrained on ImageNet)
                i.e. classes 151 to 268 of this model belong to dogs.
    :return: boolean, True if dog detected, False otherwise
    """
    predicted_idx = vgg16_predict(img_path, model)
    return 151 <= predicted_idx <= 268


def get_dog_detector_model():
    # define VGG16 model
    vgg16 = models.vgg16(pretrained=True)

    # move model to GPU if cuda is available
    if use_cuda:
        vgg16 = vgg16.cuda()
    return vgg16