"""
This module contains classes to load natural language models
and provide methods to tokenize data and get embeddings.

Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import torch
from transformers import BertTokenizer, BertModel


class BertBaseModel():
    """
    Loads a BERT model from the transformers library and provides
    wrappers for tokenizing text and getting text embeddings.
    """

    def __init__(
                self,
                max_len=45,
                model_name='bert-base-uncased',
                tensor_type='pt'
                ):
        """
        max_len: maximum number of tokens to allow
        model_name: BERT based model to load
        tensor_type: specify to use PyTorch or TensorFlow

        Load a pre-trained BERT model and matching tokenizer
        from the transformers library.
        """
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.tensor_type = tensor_type

    @torch.no_grad()
    def _tokenize(self, text):
        """
        text: string to tokenize

        Return tokens for the given text string using BERT tokenizer.
        """
        encoded_dict = self.tokenizer.encode_plus(
                                text,
                                return_tensors=self.tensor_type,
                                max_length=self.max_len,
                                padding='max_length'
                            )
        return encoded_dict['input_ids']

    @torch.no_grad()
    def _get_embedding(self, text):
        """
        text: string to get vector embedding for

        Return the BERT text embedding for the given text string.
        """
        tokens = self._tokenize(text)
        output = self.model(tokens, return_dict=True)
        return output.last_hidden_state.reshape(-1,)
