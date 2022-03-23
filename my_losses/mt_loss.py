import torch.nn as nn
import torch
import cv2
import numpy as np
from my_utils.morphology import Morphology

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.99, kernel_size=3):

        super(MultiTaskLoss, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.cross_entropy = nn.CrossEntropyLoss()
        self.erode = Morphology(1, 1, kernel_size, soft_max=False, beta=20, type='erosion2d')
        

    def forward(self, cell_predicted, boundary_predicted, target):
        target_uns = torch.unsqueeze(target,1).float()
        eroded_target = self.erode(target_uns).squeeze()
        boundary = target - eroded_target
        boundary = boundary.long()
        L_cell = self.cross_entropy(cell_predicted, target)
        L_boundary = self.cross_entropy(boundary_predicted, boundary)

        loss = self.alpha * L_cell + (1-self.alpha)*L_boundary                                                                                                                                                                                                                                                                                                                            

        return loss        

