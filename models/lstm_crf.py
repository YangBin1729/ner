__author__ = 'yangbin1729'

import pickle
import re

import numpy as np
import torch
import torch.nn as nn
from TorchCRF import CRF


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_tags = config.num_tags
        
        self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags)
    
    def forward(self, x, mask):
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats))
        outputs = self.crf.viterbi_decode(emissions, mask)
        return outputs
    
    def log_likelihood(self, x, labels, mask):
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats))
        loss = -self.crf.forward(emissions, labels, mask)
        return torch.sum(loss)


class LSTM_CRF:
    def __init__(self, config):
        with open(config.vocab_file, "rb") as inp:
            self.token2idx = pickle.load(inp)
            self.idx2token = pickle.load(inp)
        
        self.tag2idx = {'E_nt': 0, 'M_nt': 1, 'E_nr': 2, 'O': 3,
                        'B_nr': 4, 'B_ns': 5, 'M_ns': 6,
                        'E_ns': 7, 'M_nr': 8, 'B_nt': 9}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        
        config.vocab_size = len(self.token2idx)
        config.num_tags = len(self.tag2idx)
        
        self.model = Model(config)
        self.model.load_state_dict(torch.load(config.bilstm_crf_file, map_location=torch.device(
            'cpu')))
        self.config = config
    
    def encode(self, text, maxlen):
        seqs = re.split('[，。！？、‘’“”:]', text)
        seq_ids = []
        processed_text = ''
        for sentence in seqs:
            token_ids = []
            for char in sentence:
                if char not in self.token2idx:
                    token_ids.append(self.token2idx['[unknown]'])
                else:
                    token_ids.append(self.token2idx[char])
            seq_ids.append(token_ids)
            processed_text += sentence
            
        num_samples = len(seq_ids)
        x = np.full((num_samples, maxlen), 0., dtype=np.int64)
        for idx, s in enumerate(seq_ids):
            trunc = np.array(s[:maxlen], dtype=np.int64)
            x[idx, :len(trunc)] = trunc
        return x, processed_text
    
    def predict(self, text):
        x, processed_text = self.encode(text, maxlen=self.config.maxlen)
        input_ids = torch.tensor(x)
        mask = input_ids > 0
        
        self.model.eval()
        paths = self.model(input_ids, mask)
        
        tags = []
        for path, m in zip(paths, mask):
            length = sum(m)
            tags.extend([self.idx2tag[idx] for idx in path[:length]])
        
        return [tag if tag == 'O' else tag[-2:] for tag in tags], processed_text