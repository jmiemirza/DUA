import torch.nn as nn

from utils.rotation import *


def test(dataloader, model, sslabel=None):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    model.eval()
    correct = []
    losses = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if sslabel is not None:
            inputs, labels = rotate_batch(inputs, sslabel)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    return 1- correct.mean(), correct, losses
