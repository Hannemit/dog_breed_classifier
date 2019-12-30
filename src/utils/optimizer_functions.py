import torch.optim as optim


def optimizer_adam(model_params, learning_rate=0.001):
    return optim.Adam(model_params, lr=learning_rate)
