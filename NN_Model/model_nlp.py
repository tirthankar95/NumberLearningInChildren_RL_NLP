import torch 
import torch.nn as nn 
import numpy as np
from torch.distributions import Categorical 
import model_strategy_if as MS 
# FROM NLTK, download takes lot of time.
stopwords = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", \
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', \
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", \
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', \
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', \
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', \
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', \
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', \
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', \
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', \
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', \
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', \
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', \
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', \
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', \
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', \
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', \
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
]

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
        self.token_dict = {}; self.token_dict['[PAD]'] = 0
        self.token_cnt = 1
    
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
                if word in stopwords: continue
                if word in self.punctuations: continue
                elif word[-1] == '.' or word[-1] == ',':
                    word = word[:-1]
                elif word[-2:] == "\'s":
                    word = word[:-2]
                if word not in self.token_dict:
                    self.token_dict[word] = self.token_cnt
                    self.token_cnt += 1
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
    
