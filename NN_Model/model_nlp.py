import torch 
import torch.nn as nn 
import numpy as np
from torch.distributions import Categorical 
import model_strategy_if as MS 
import json



class NN_Simple(MS.NN_Strategy):
    def __init__(self):
        super().__init__()
        self.punctuations = ['.', ',', '!', '?', '*']
        self.mxSentenceLength, self.noOfActions= 32, 6
        self.fc = nn.Sequential( nn.Linear(self.mxSentenceLength, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32)

        )
        # ACTOR
        self.actor = nn.Sequential( nn.ReLU(),
                                       nn.Linear(32, self.noOfActions)
        )
        # CRITIC
        self.critic = nn.Sequential( nn.ReLU(),
                                        nn.Linear(32, 1)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open("./NN_Model/model_data/token.json", "r") as file:
            self.token_dict = json.load(file)
        # FROM NLTK, download takes lot of time.
        with open("./NN_Model/model_data/stopwords.json", "r") as file:
            self.stopwords = json.load(file)

    
    def pre_process(self, state):
        xs = [state["text"]]
        n = len(xs)
        sen_arr = [[] for i in range(n)]
        for id_out, x in enumerate(xs):
            sen_x = x.lower().split()
            new_sen_x = []
            # Tokenize & remove useless words.
            for id, word in enumerate(sen_x):
                word = word.lower()
                if word in self.stopwords: continue
                if word in self.punctuations: continue
                elif word[-1] == '.' or word[-1] == ',': word = word[:-1]
                elif word[-2:] == "\'s": word = word[:-2]
                new_sen_x.append(self.token_dict[word])
            # Padding & Truncation.
            if len(new_sen_x) < self.mxSentenceLength:
                # 0 ~ is padding.
                new_sen_x = new_sen_x + [0] * (self.mxSentenceLength - len(new_sen_x))
            elif len(new_sen_x) > self.mxSentenceLength:
                new_sen_x = new_sen_x[:self.mxSentenceLength]
            sen_arr[id_out] = new_sen_x 
            state["text"] = torch.tensor(sen_arr, dtype = torch.float32)
        return state
    
    def rearrange(self, state):
        txt = []
        for st in state:
            txt.append(st["text"].squeeze(0))
        return torch.stack(txt)
    
    def forward(self, state):
        xtr = self.rearrange(state)
        if torch.cuda.is_available():
            text = {key: val.to('cuda:0') for key, val in text.items()}
        op = self.fc(xtr)
        if torch.cuda.is_available(): 
            op = op.to(torch.device("cuda:0")) 
        mu_dist = Categorical(logits = self.actor(op))
        value = self.critic(op)
        return mu_dist, value 
    
