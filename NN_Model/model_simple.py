import torch 
import torch.nn as nn 
import numpy as np
from torch.distributions import Categorical 

class NN_Simple(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.freq_words = ['this', 'the', 'is', 'are', 'you', 'our', 'in']
        self.token_dict = {}; self.token_dict['[PAD]'] = 0
        self.token_cnt = 1
    
    def pre_process(self, xs):
        n = len(xs)
        sen_arr = [[] for i in range(n)]
        for id_out, x in enumerate(xs):
            sen_x = x.lower().split()
            new_sen_x = []
            # Tokenize & remove useless words.
            for id, word in enumerate(sen_x):
                if word in self.freq_words: continue 
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
        return torch.tensor(sen_arr, dtype = torch.float32)

    def forward(self, xtr):
        # xtr = self.pre_process(x)
        if torch.cuda.is_available():
            text = {key: val.to('cuda:0') for key, val in text.items()}
        op = self.fc(xtr)
        if torch.cuda.is_available(): 
            op = op.to(torch.device("cuda:0")) 
        mu_dist = Categorical(logits=self.actor(op))
        value = self.critic(op)
        return mu_dist, value 
    
# if __name__ == '__main__':
#     nn_obj = NN_Simple()
#     xtr = ["This is one hundred twenty . There are 1 blocks in hundred's place, 2 blocks in ten's place and 0 blocks in unit's place.",
#            "This is one hundred twenty . There are 0 blocks in hundred's place, 0 blocks in ten's place and 0 blocks in unit's place. You are holding a hundred block."]
#     mu, val = nn_obj(xtr)
#     print(mu.sample())
