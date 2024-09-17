import torch 
import torch.nn as nn 
import torchvision.models as models 
import model_strategy_if as MS 
import numpy as np
from transformers import BertModel, BertTokenizer
from torch.distributions import Categorical
from collections import defaultdict
import logging
LOG = logging.getLogger(__name__)

noOfActions=6
epochA=0
class NNModelNLP(MS.NN_Strategy):
    def __init__(self, job_type = None):
        super().__init__()
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
            param.requires_grad = False 
        layerNotTrainable = 'layer3'
        for i, (name, param) in enumerate(self.resnet.named_parameters()):
            if name[:len(layerNotTrainable)]=='layer4':
                param.requires_grad = True
        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.mxSentenceLength = 50
        for param in self.bert.parameters():
            param.requires_grad=False 
        self.bert_hidden=self.bert.config.hidden_size
        self.bertfc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bert_hidden*self.mxSentenceLength,(self.bert_hidden*self.mxSentenceLength)//100),
            nn.ReLU(),
            nn.Linear((self.bert_hidden*self.mxSentenceLength)//100,(self.bert_hidden*self.mxSentenceLength)//200)
        )
        self.bertfinLength=(self.bert_hidden*self.mxSentenceLength)//200
        # DQN
        self.noOfActions=6
        self.actor=nn.Sequential(
            nn.Linear(self.bertfinLength+self.important_features_image,self.important_features_image//2),
            nn.ReLU(),
            nn.Linear(self.important_features_image//2,self.noOfActions)
        )        
        self.critic=nn.Sequential(
            nn.Linear(self.bertfinLength+self.important_features_image,self.important_features_image//2),
            nn.ReLU(),
            nn.Linear(self.important_features_image//2,1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # DQN for transfer task
        if job_type == 'transfer_task':
            self.transfer = nn.Sequential(
                                            nn.Linear(self.noOfActions, 3), # Transfer task has 3 output (greater, equal, lesser)                                        nn.Softmax(dim = 1)
                                        )
    def make_trainable(self, option):
        if option == "full" or option == "vision":
            for param in self.resnet.parameters():
                param.requires_grad = True 
        if option == "full" or option == "language":
            for param in self.bert.parameters():
                param.requires_grad = True 

    def rearrange(self, state):
        imgs, txt = [], defaultdict(list)
        for st in state:
            imgs.append(st["visual"].squeeze(0))
            for k, v in st["text"].items():
                txt[k].append(v.squeeze(0))
        for k, v in txt.items():
            txt[k] = torch.stack(txt[k])
        return torch.stack(imgs), txt
    
    def forward(self, state):
        image, text = self.rearrange(state)
        image = image.to(self.device)
        op = self.resnet(image)
        opR1 = self.fcResNet1(op)
        if torch.cuda.is_available():
            text = {key: val.to('cuda:0') for key, val in text.items()}
        opR2 = self.bert(**text)[0]
        opR2 = self.bertfc(opR2)
        if torch.cuda.is_available():
            opR1 = opR1.to(torch.device("cuda:0")) 
            opR2 = opR2.to(torch.device("cuda:0")) 
        mu_dist = Categorical(logits=self.actor(torch.cat([opR2,opR1],dim=1)))
        value = self.critic(torch.cat([opR2,opR1],dim=1))
        return mu_dist,value 
    
    def pre_process(self, state):
        # VISUAL
        state["visual"] = torch.FloatTensor(np.array([state["visual"]]))
        temp_vis = torch.squeeze(state["visual"])
        temp_vis = temp_vis.transpose(0,1).transpose(0,2) 
        state["visual"] = torch.unsqueeze(temp_vis, dim=0)
        # TEXTUAL
        TEXT = state["text"]+" [PAD]" * self.mxSentenceLength 
        state["text"] = self.tokenizer(TEXT, padding = True, truncation = True, \
                                       max_length = self.mxSentenceLength,return_tensors="pt")
        return state 
    
    def display(self):
        for param in self.fcResNet0:
            print(param)
