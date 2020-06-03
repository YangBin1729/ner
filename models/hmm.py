__author__ = 'yangbin1729'

import numpy as np
import pickle, re


class HMM:
    def __init__(self, config):
        with open(config.vocab_file, "rb") as inp:
            self.token2idx = pickle.load(inp)
            self.idx2token = pickle.load(inp)
        
        self.tag2idx = {'E_nt': 0, 'M_nt': 1, 'E_nr': 2, 'O': 3,
                        'B_nr': 4, 'B_ns': 5, 'M_ns': 6,
                        'E_ns': 7, 'M_nr': 8, 'B_nt': 9}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        
        with open(config.hmm_file, "rb") as inp:
            self.start_status = pickle.load(inp)
            self.emissions = pickle.load(inp)
            self.transitions = pickle.load(inp)
    
    def viterbi_decode(self, x):
        T = len(x)
        N = len(self.tag2idx)
        
        dp = np.full((T, N), float('-inf'))
        ptr = np.zeros_like(dp, dtype=np.int32)
        
        dp[0] = self.log_(self.start_status) + self.log_(self.emissions[:, x[0]])
        
        for i in range(1, T):
            v = dp[i - 1].reshape(-1, 1) + self.log_(self.transitions)
            dp[i] = np.max(v, axis=0) + self.log_(self.emissions[:, x[i]])
            ptr[i] = np.argmax(v, axis=0)
        
        best_seq = [0] * T
        best_seq[-1] = np.argmax(dp[-1])
        for i in range(T - 2, -1, -1):
            best_seq[i] = ptr[i + 1][best_seq[i + 1]]
        
        return best_seq
    
    @staticmethod
    def log_(v):
        return np.log(v + 0.000001)
    
    def predict(self, text):
        res = []
        start = 0
        processed_text = ''
        for m in re.finditer('[，。！？、‘’“”:]', text):
            end = m.span()[0]
            seg = text[start:end]
            if seg:
                seg_ids = [self.token2idx[token] if token in self.token2idx else self.token2idx[
                    '[unknown]'] for token in seg]
                seg_labels = self.viterbi_decode(seg_ids)
                res.extend(seg_labels)
                processed_text += seg
            start = m.span()[1]
        
        if start != len(text):
            seg = text[start:]
            if seg:
                seg_ids = [self.token2idx[token] if token in self.token2idx else self.token2idx[
                    '[unknown]'] for token in seg]
                seg_labels = self.viterbi_decode(seg_ids)
                res.extend(seg_labels)
                processed_text += seg
        
        tags = [self.idx2tag[idx] for idx in res]
        return [tag if tag == 'O' else tag[-2:] for tag in tags], processed_text