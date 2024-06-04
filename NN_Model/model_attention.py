import logging 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Categorical
import os 

# log_level = os.getenv('LOGLEVEL', 'INFO')
# logging.basicConfig(level=getattr(logging, log_level.upper(), None))
LOG = logging.getLogger(__name__)
__author__ = 't.mittra'

class ScaledDotProduct(nn.Module):
    def __init__(self, temperature = 1, attn_dropout=0.1):
        super().__init__()
        self.temp = temperature
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, k, q, v):
        attn = torch.matmul(q/self.temp, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        logging.debug(f'Shape of attention {attn.shape}')
        return output, attn

class AttentionLayer(nn.Module):
    def __init__(self, dmodel, dkq, dv, nhead, attn_dropout = 0.1):
        super().__init__()
        self.dmodel = dmodel
        self.dkq = dkq
        self.dv = dv
        self.nhead = nhead 
        self.wk = nn.Linear(self.dmodel, self.nhead*self.dkq, bias = False)
        self.wq = nn.Linear(self.dmodel, self.nhead*self.dkq, bias = False)
        self.wv = nn.Linear(self.dmodel, self.nhead*self.dv, bias = False)
        self.attention = ScaledDotProduct(self.dkq**0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(self.dmodel, eps = 1e-6) 
        self.fc = nn.Linear(self.nhead*self.dv, self.dmodel)

    def forward(self, ip):
        m, tokens_in_sentence = ip.shape[0], ip.shape[1]
        residual = ip
        k = self.wk(ip).view(m, tokens_in_sentence, self.nhead, self.dkq)
        q = self.wq(ip).view(m, tokens_in_sentence, self.nhead, self.dkq)
        v = self.wv(ip).view(m, tokens_in_sentence, self.nhead, self.dv) 
        
        # m * tokens_in_sentence * nhead * dkq -> m * nhead * tokens_in_snetence * dkq
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(k, q, v)
        # m * nhead * tokens_in_snetence * dkq -> m * tokens_in_sentence * nhead * dkq
        '''
        Contiguous: .contiguous() is then called to ensure that the transposed tensor is contiguous in memory. 
        This is necessary because certain operations, like view (reshaping), expect the tensor to be contiguous.
        '''
        q = q.transpose(1, 2).contiguous().view(m, tokens_in_sentence, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class NNAttention(nn.Module):
    # mx_sen_len has to be a mulitple of 2 and less than 1024.
    def __init__(self, img_shape, embedding_val = 50, mx_sen_len = 32):
        super().__init__()
    # CNN
        cnn_channels = [16, 32, 64]
        self.cnn0 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = cnn_channels[0],
                                            kernel_size = 3, padding = 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features = cnn_channels[0]),
                                  nn.MaxPool2d(kernel_size = 2, stride = 2)
                                )
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels = cnn_channels[0], 
                                            out_channels = cnn_channels[1],
                                            kernel_size = 3, padding = 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features = cnn_channels[1]),
                                  nn.MaxPool2d(kernel_size = 2, stride = 2)
                                )
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels = cnn_channels[1], 
                                            out_channels = cnn_channels[2],
                                            kernel_size = 3, padding = 1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features = cnn_channels[2]),
                                  nn.MaxPool2d(kernel_size = 2, stride = 2)
                                )
        dummy_m = 2
        dummy_img = torch.rand(dummy_m,*img_shape)
        dummy_out_img = self.cnn2(self.cnn1(self.cnn0(dummy_img))).view(dummy_m, -1)
        self.img_op = 128
        self.dnn = nn.Sequential(nn.Linear(dummy_out_img.shape[-1], self.img_op*2),
                                 nn.ReLU(),
                                 nn.Linear(self.img_op*2, self.img_op))
    # NLP ATTENTION
        self.mx_sen_len = mx_sen_len
        ## GLOVE EMBEDDING.
        self.embedding = {}
        if embedding_val not in [50, 100, 200, 300]:
            LOG.error(f'wrong embedding value.')
            assert False
        with open(f'glove.6B.{embedding_val}d.txt', 'r') as file:
            for line in file:
                list_ = line.split()
                self.embedding[list_[0]] = [float(list_[x]) for x in range(1, embedding_val+1)]
        self.embedding["[PAD]"] = [0.0]* embedding_val
        self.dmodel0 = embedding_val
        self.dkq0 = 64
        self.dv0 = 64
        self.nhead0 = 16
        self.attn0 = AttentionLayer(self.dmodel0, self.dkq0, self.dv0, self.nhead0)
    # COMBINED ATTENTION
        self.dmodel1 =   self.img_op//self.mx_sen_len + self.dmodel0
        self.dkq1 = 64
        self.dv1 = 64
        self.nhead1 = 16
        self.attn1 = AttentionLayer(self.dmodel1, self.dkq1, self.dv1, self.nhead1)
    # ACTOR
        self.noOfActions = 6
        self.actor=nn.Sequential(
            nn.Linear(self.dmodel1*self.mx_sen_len, self.dmodel1),
            nn.ReLU(),
            nn.Linear(self.dmodel1,self.noOfActions)
        )        
    # CRITIC
        self.critic=nn.Sequential(
            nn.Linear(self.dmodel1*self.mx_sen_len, self.dmodel1),
            nn.ReLU(),
            nn.Linear(self.dmodel1,1)
        )

    def forward(self, imgs, p_text_t):
        tr_ex, txt_len = imgs.shape[0], self.mx_sen_len
        ip0 = self.cnn2(self.cnn1(self.cnn0(imgs)))
        ip0 = ip0.view(tr_ex, -1) 
        ip1 = self.dnn(ip0)
        ip1 = ip1.view(tr_ex, txt_len, -1)
        ip2, attn0 = self.attn0(p_text_t)
        if torch.cuda.is_available():
            ip1 = ip1.to(torch.device("cuda:0")) 
            ip2 = ip2.to(torch.device("cuda:0")) 
        logging.debug(f'Shape of output image[{ip1.shape}] & text[{ip2.shape}]')
        ip3 = torch.cat([ip1, ip2], dim = -1)
        logging.debug(f'Shape of concat input [{ip3.shape}]')
        op, attn1 = self.attn1(ip3)
        op = op.view(tr_ex, -1)
        logging.debug(f'Shape of final output [{op.shape}]')
        mu_dist = Categorical(logits = self.actor(op))
        value = self.critic(op)
        return mu_dist, value
    
    def pre_process(self, texts):
    # Preprocess text.
        tr_ex, txt_len = len(texts), self.mx_sen_len
        p_text = [[[] for _ in range(txt_len)] for __ in range(tr_ex)]
        for idxx, text in enumerate(texts):
            sen = []
            for word in text.split():
                word = word.lower()
                if word == ',' or word == '.': continue
                elif word[-1] == '.' or word[-1] == ',':
                    word = word[:-1]
                elif word[-2:] == "\'s":
                    word = word[:-2]
                sen.append(word)
            if len(sen) < txt_len:
                sen = sen + ['[PAD]'] * (txt_len - len(sen))
            elif len(sen) > txt_len:
                logging.error('Increase mx sentence length.')
                assert False
            for idx, word in enumerate(sen):
                p_text[idxx][idx] = self.embedding[word]
        p_text_t = torch.tensor(p_text)
        logging.debug(f'Shape of input text {p_text_t.shape}')
        return p_text_t
    
# if __name__ == '__main__':
#     tr_ex, img_shape = 2, [3, 224, 224]
#     img = torch.rand(tr_ex, *img_shape)
#     nn = NNAttention(img_shape)
#     text = ["This is one. Let's use our blocks to build the number.",
#     "This is one. There are 0 blocks in hundred's place, 0 blocks in ten's place and 0 blocks in unit's place. You are holding a unit block."]
#     dist, value = nn(img, nn.pre_process(text))
#     print(dist.sample().shape, value.shape)
#     # print()
