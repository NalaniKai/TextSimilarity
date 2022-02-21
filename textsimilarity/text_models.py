"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

class BertBaseModel():
    def __init__(self, model_name='bert-base-uncased', tensor_type='pt'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name) 
        self.model = BertModel.from_pretrained(model_name)
        self.tensor_type = tensor_type

    def _tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors=self.tensor_type)

    def get_embedding(self, text):
        tokens = self._tokenize(text)
        output = self.model(tokens, return_dict=True)
        return output.last_hidden_state