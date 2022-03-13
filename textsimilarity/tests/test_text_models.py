"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import unittest
from textsimilarity import text_models
from textsimilarity import constants as c


class TestBertModel(unittest.TestCase):
    """
    Test methods in BertBaseModel for tokenizing
    text and getting text embeddings.
    """

    def setUp(self):
        """Define test examples for text phrases."""
        self.bert_model = text_models.BertBaseModel()
        self.test_park_walk = 'walk in the park'
        self.test_out_walk = 'out for a walk'

    def tearDown(self):
        """Reset setup"""
        self.bert_model = None
        self.test_park_walk = None
        self.test_out_walk = None

    def test__tokenize_not_equal(self):
        """Different words should have different tokens."""
        token_park_walk = self.bert_model._tokenize(self.test_park_walk)
        token_out_walk = self.bert_model._tokenize(self.test_out_walk)

        token_walk = token_park_walk[c.FIRST_IDX][c.SECOND_IDX]
        token_out = token_out_walk[c.FIRST_IDX][c.SECOND_IDX]

        self.assertNotEqual(token_walk, token_out)

    def test__tokenize_equal(self):
        """The same words should have the same tokens."""
        token_park_walk = self.bert_model._tokenize(self.test_park_walk)
        token_out_walk = self.bert_model._tokenize(self.test_out_walk)

        words_in_out_walk = len(self.test_out_walk.split())
        token_walk_from_park = token_park_walk[c.FIRST_IDX][c.SECOND_IDX]
        token_walk_from_out = token_out_walk[c.FIRST_IDX][words_in_out_walk]
        self.assertEqual(token_walk_from_park, token_walk_from_out)

    def test__get_embedding(self):
        """Embeddings should be of expected size."""
        emb = self.bert_model._get_embedding(self.test_out_walk)
        self.assertEqual(emb.shape[c.FIRST_IDX], c.EMB_SIZE)
