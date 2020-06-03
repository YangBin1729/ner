__author__ = 'yangbin1729'

from config import Config
from .lstm_crf import LSTM_CRF
from .hmm import HMM
from .albert import ALBERT


def predict(text):
    config = Config()
    outputs = {}
    
    ####### LSTM_CRF 模型 ##################################################
    model = LSTM_CRF(config)
    tags, processed_text = model.predict(text)
    outputs["LSTM_CRF"] = list(zip(processed_text, tags))
    
    ####### HMM 模型 #######################################################
    hmm = HMM(config)
    tags, processed_text = hmm.predict(text)
    outputs["HMM"] = list(zip(processed_text, tags))
    
    ####### ALBERT 模型 ####################################################
    albert = ALBERT(config)
    tags, processed_text = albert.predict(text)
    outputs['ALBERT'] = list(zip(processed_text, tags))
    
    return outputs

