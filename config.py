__author__ = 'yangbin1729'


class Config:
    
    vocab_file = "saved/vocab.pkl"
    tags_file = "saved/tags.pkl"
    
    ########### BiLSTM-CRF模型相关参数 ##########################
    embedding_dim = 100
    hidden_dim = 200
    dropout = 0.2
    
    maxlen = 60
    
    bilstm_crf_file = "saved/params.pkl"
    
    ########### HMM模型相关参数 #################################
    hmm_file = "saved/hmm.pkl"
    
    
    ########### HMM模型相关参数 #################################
    albert_path = "saved/albert"

        