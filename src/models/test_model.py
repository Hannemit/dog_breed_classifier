import numpy as np
from src.data import data_loader
from src.models import cnn_from_scratch
from src.utils import criterion_functions
import sys
import torch


def test(model, criterion, use_cuda: bool, test_data_path: str = ""):
    """
    Test our model, print out loss and accuracy on test data
    :param model: trained model
    :param criterion: pytorch loss function
    :param use_cuda: boolean, move to GPU or not
    :param test_data_path: string, path to where the test data is situated.
    :return:
    """
    # get loader for test data
    loader = data_loader.get_test_loader(path=test_data_path)

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print(f'Test Loss: {test_loss:.6f}')
    print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError("Supply a single argument, the path to a model you want to test, e.g. call "
                         "'python src/models/test_model.py models/my_model_100.pt'")
    model_path = args[0]
    use_cuda = torch.cuda.is_available()
    criterion = criterion_functions.cross_entropy_loss()
    model = cnn_from_scratch.Net()

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"No model found at {model_path}. Train a model first or "
                                f"put an already-trained model there")

    print(f"Testing model {model_path} with cross entropy loss")
    test(model, criterion, use_cuda)


if __name__ == "__main__":
    main()
