import torch
import torch.nn as nn

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, inputs, targets):
        mask = (inputs != 0) & (targets != 0)
        loss = torch.abs(inputs - targets)
        masked_loss = loss[mask]
        return masked_loss.mean()