import numpy as np
import torch


def train(n_epochs: int, loaders, model, optimizer, criterion, use_cuda: bool, save_path: str):
    """
    Return a trained model
    :param n_epochs: int, number of epochs to train for
    :param loaders: dict, pytorch data loaders, should have keys "train", "test" and "valid", with pytorch
                    loaders as values
    :param model: an instance of our CNN class
    :param optimizer: a pytorch optimizer
    :param criterion: a pytorch loss function
    :param use_cuda: boolean, if True use GPU to train
    :param save_path: string, path where to save best model.
    :return: The trained model. The best model is saved at save_path.
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        print(f"Starting epoch {epoch}")
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train
        print("\tTraining..")
        model.train()
        for batch_idx, (data, target) in enumerate(loaders["train"]):
            if use_cuda:
                # move to GPU
                data, target = data.cuda(), target.cuda()
            output = model(data)

            # reset gradients
            optimizer.zero_grad()

            # loss and backward pass
            loss = criterion(output, target)
            loss.backward()

            # update weights
            optimizer.step()

            # keep track of loss
            train_loss = train_loss + ((1.0 / (batch_idx + 1)) * (loss.data - train_loss))

        # validate model
        print("\tEvaluating..")
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders["valid"]):
            if use_cuda:
                # move to GPU
                data, target = data.cuda(), target.cuda()

            # update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1.0 / (batch_idx + 1)) * (loss.data - valid_loss))

        # print training/validation statistics
        print(f"\tEpoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}")

        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f"\tValidation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return trained model
    return model
