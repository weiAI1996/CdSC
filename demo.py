import os
import glob
import cv2
import shutil
import random
import numpy as np
import torch
#读取txt
loss_f = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
A = torch.tensor([[[1,0],[1,1]]]).float()
B = torch.tensor([[[0,1],[0,0]]]).float()

print(loss_f(A.view(1,-1),B.view(1,-1)))

