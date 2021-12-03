import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, targets):
        # Masking into a vector of 1's and 0's.
        mask = (targets != 0)
        mask = mask.float()

        # Actual number of ratings.
        # Take max to avoid division by zero while calculating loss.
        other = torch.Tensor([1.0])
        other = other.cuda()
        number_ratings = torch.max(torch.sum(mask), other)
        error = torch.sum(torch.mul(mask, torch.mul((targets - inputs), (targets - inputs))))
        loss = error.div(number_ratings)
        return loss[0]
