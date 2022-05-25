from math import e
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda, FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader

import datetime
import sys
import requests
import io
import os
import random
from time import sleep

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics


#Plotting
def get_sample_points(G, z, n_dim = 2):
    """
      Generate 'size' data samples
      ___

      G = Generator (in G.eval() mode)
      z = Random noise tensor of shape (n, 2, 1)
      n = number of points
    """
    z_gen,_ = G(z)
    z_gen = z_gen.permute(2,0,1)
    n = z_gen.shape[1]
    batch_size = z_gen.shape[0]
    y_hat = z_gen.reshape(n * batch_size, n_dim) 
    points = y_hat.cpu().data.numpy()
    return points

#Probability calculator with privacy budget
def dp_proba(eps, d):
  p = (e**eps) / (e**eps + d - 1)
  q = 1 / (e**eps + d - 1)
  return p, q