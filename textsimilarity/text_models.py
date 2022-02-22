"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

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
        Load a pre-trained BERT model and matching tokenizer
        from the transformers library.
        """
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(model_name) 
        self.model = BertModel.from_pretrained(model_name)
        self.tensor_type = tensor_type

    def _tokenize(self, text):
        """Returned the tokens for the given text using BERT tokenizer."""
        encoded_dict = self.tokenizer.encode_plus(text, 
        return_tensors=self.tensor_type,
        max_length=self.max_len,
        padding='max_length')
        return encoded_dict['input_ids']

    def _get_embedding(self, text):
        """Return the BERT text embedding for the given text."""
        tokens = self._tokenize(text)
        output = self.model(tokens, return_dict=True)
        return output.last_hidden_state.reshape(-1,)