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

###
# Large Spatial Transformer Network (STN) module
# 
# Input:
#   k = number of point dimensions
###
class STNkd(nn.Module):
    def __init__(self, k=2):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = F.relu(self.bn8(self.fc3(x)))
        x = self.fc4(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x

###
# PointNet encoder
#
# Input:
#   code_nfts = number of point embedding dimensions
#   num_points = number of points 
#   n_dim = number of point dimensions
#   global_feat = if True, return global feature embedding, if False, return individual (per-point) embeddings
#   trans = if True, use TransformNet implementation, if False, use Vanilla PointNet
###
class PointNetfeat(nn.Module):
    """
        PointNet Module
    """
    def __init__(self, code_nfts=2048, num_points=2500, n_dim = 2, global_feat=True, trans=True):
        super(PointNetfeat, self).__init__()
        self.n_dim = n_dim
        self.stn = STNkd(k = n_dim)
        self.code_nfts = code_nfts
        self.conv1 = torch.nn.Conv1d(n_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, code_nfts, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(code_nfts)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.code_nfts)
        if self.trans:
            if self.global_feat:
                return x #, trans
            else:
                x = x.view(-1, self.code_nfts, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

###
# PointNet Generator
#
# Input:
#   code_nfts = number of point embedding dimensions
#   n_dim = number of point_dimensions
#   global_feat = argument for PointNet: if True, return global feature embedding, if False, return individual (per-point) embeddings
#   trans = argument for PointNet: if True, use TransformNet implementation, if False, use Vanilla PointNet
###
class PointNet_Generator(nn.Module):
    """ 
        Generator with PointNet Encoder, MLP Decoder
    """
    def __init__(self, code_nfts=2048, n_dim=2, global_feat=True, trans=True):
        super(PointNet_Generator, self).__init__()
        self.code_nfts = code_nfts
        self.n_dim = n_dim
        self.encoder = nn.Sequential(
            PointNetfeat(code_nfts, 1, n_dim = n_dim, global_feat=global_feat, trans=trans),
            nn.Linear(code_nfts, code_nfts),
            nn.BatchNorm1d(code_nfts),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_nfts, code_nfts),
            nn.BatchNorm1d(code_nfts),
            nn.ReLU(),
            nn.Linear(code_nfts, code_nfts // 2),
            nn.BatchNorm1d(code_nfts // 2),
            nn.ReLU(),
            nn.Linear(code_nfts // 2, n_dim),
            nn.Tanh()
        )


    def forward(self, x):
        #Encoder
        code = self.encoder(x)
        #Decoder
        x = self.decoder(code)
        x = x.view(-1, self.n_dim, 1)
        return x, code

###
# PointNet Discriminator
#
# Input:
#   code_nfts = number of point embedding dimensions
#   n_dim = number of point dimensions
#   global_feat = argument for PointNet: if True, return global feature embedding, if False, return individual (per-point) embeddings
#   trans = argument for PointNet: if True, use TransformNet implementation, if False, use Vanilla PointNet
###
class PointNet_Discriminator(nn.Module):
    """ 
        PointNet Discriminator
    """
    def __init__(self, code_nfts=2048, n_dim = 2, global_feat=True, trans = False):
        super(PointNet_Discriminator, self).__init__()
        self.n_dim = n_dim
        self.code_nfts = code_nfts
        self.cls = nn.Sequential(
            PointNetfeat(code_nfts, 1, n_dim = n_dim, global_feat=global_feat, trans=trans),
            nn.Linear(code_nfts, code_nfts),
            nn.BatchNorm1d(code_nfts),
            nn.ReLU(),
            nn.Linear(code_nfts, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #Point level classifier
        x = self.cls(x)
        return x