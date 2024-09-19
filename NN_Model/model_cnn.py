import torch 
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from torch.distributions import Categorical
import utils as U
import json
import copy 

noOfActions=6
epochA=0
class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18
        self.resnet=models.resnet18(pretrained=True)
        num_dim_in=self.resnet.fc.in_features
        
        num_dim_out=1024
        self.resnet.fc=nn.Linear(num_dim_in,num_dim_out)
        self.important_features_image=512
        self.fcResNet1=nn.Sequential(
            nn.Linear(num_dim_out,self.important_features_image),
            nn.ReLU()
        )
        for param in self.resnet.parameters():
            param.requires_grad=False 
        layerNotTrainable='layer4'
        for i, (name, param) in enumerate(self.resnet.named_parameters()):
            if name[:len(layerNotTrainable)]=='layer4':
                param.requires_grad=True
        # DQN
        self.noOfActions=6
        self.actor=nn.Sequential(
            nn.Linear(self.important_features_image,self.important_features_image//2),
            nn.ReLU(),
            nn.Linear(self.important_features_image//2,self.noOfActions)
        )
        self.critic=nn.Sequential(
            nn.Linear(self.important_features_image,self.important_features_image//2),
            nn.ReLU(),
            nn.Linear(self.important_features_image//2,1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self,image):
        image = image.to(self.device)
        op=self.resnet(image)
        opR1=self.fcResNet1(op)
        mu_dist=Categorical(logits=self.actor(opR1))
        value=self.critic(opR1)
        return mu_dist,value

    def display(self):
        for param in self.fcResNet0:
            print(param)
'''
------------------------------------------------------------------------------------
'''
