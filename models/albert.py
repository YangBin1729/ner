__author__ = 'yangbin1729'

import torch
from transformers import BertTokenizer, AlbertForTokenClassification


class ALBERT:
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained(config.albert_path)
        self.model = AlbertForTokenClassification.from_pretrained(config.albert_path)
        
        tags = [
            'B_ns', 'M_ns', 'E_ns', 'B_nr', 'M_nr', 'E_nr', 'B_nt', 'M_nt', 'E_nt',
            'O',
        ]
        self.tag2id = {tag: idx for idx, tag in enumerate(tags)}
        self.id2tag = {idx: tag for idx, tag in enumerate(tags)}
    
    def predict(self, text):
        self.model.eval()
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False)).unsqueeze(0)
        processed_text = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        outputs = self.model(input_ids)
        prediction_scores = outputs[0]
        pred = torch.argmax(prediction_scores, dim=2).squeeze(0)
        tags = [self.id2tag[idx] for idx in pred.tolist()]
        tags = [tag if tag=='O' else tag[-2:] for tag in tags]

        return tags, processed_text