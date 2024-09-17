import torch 
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch.optim as optim
import model_strategy_if as MS 
import numpy as np
import matplotlib.pyplot as plt 
from torch.distributions import Categorical
import utils as U
import json
import copy 

noOfActions=6
epochA=0
class NNModel(MS.NN_Strategy):
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

    def rearrange(self, state):
        imgs = []
        for st in state:
            imgs.append(st["visual"].squeeze(0))
        return torch.stack(imgs)
    
    def forward(self, state):
        image = self.rearrange(state)
        image = image.to(self.device)
        op=self.resnet(image)
        opR1=self.fcResNet1(op)
        mu_dist=Categorical(logits=self.actor(opR1))
        value=self.critic(opR1)
        return mu_dist,value
    
    def pre_process(self, state):
        state["visual"] = torch.FloatTensor(np.array([state["visual"]]))
        temp_vis = torch.squeeze(state["visual"])
        temp_vis = temp_vis.transpose(0,1).transpose(0,2) 
        state["visual"] = torch.unsqueeze(temp_vis, dim=0)
        return state
    
    def display(self):
        for param in self.fcResNet0:
            print(param)
