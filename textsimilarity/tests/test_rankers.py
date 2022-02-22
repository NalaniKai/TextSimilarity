"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import unittest 
from textsimilarity import rankers
from textsimilarity import text_models
from textsimilarity import constants as c

class TestRankers(unittest.TestCase):
    """
    Test methods for CosineSimilarityRanker for 
    getting text embeddings, calculating the cosine 
    similarity, and ranking text based on similarity.
    """

    def setUp(self):
        """
        Define test instances and text examples.
        """
        self.bert_model = text_models.BertBaseModel()
        self.comparison_corpus = ['happy birthday', 'party invitations', 
                'special menu', 'wedding party']
        self.cosine_sim_ranker = rankers.CosineSimilarityRanker(
                                        self.bert_model, 
                                        self.comparison_corpus
                                        )

    def tearDown(self):
        """Reset setup"""
        self.bert_model = None
        self.comparison_corpus = None 
        self.cosine_sim_ranker = None

    def test__get_embeddings_dict_text_compare(self):
        """
        The keys in the comparision dictionary should match
        the input text.
        """
        emb_dict = self.cosine_sim_ranker._get_embeddings_dict(
                                        self.comparison_corpus)
        self.assertEqual(list(emb_dict.keys()), self.comparison_corpus)

    def test__get_embeddings_dict_emb_compare(self):
        """
        The embeddings in the comparison dictionary should be of
        expected size.
        """
        emb_dict = self.cosine_sim_ranker._get_embeddings_dict(
                                        self.comparison_corpus)
        embs = list(emb_dict.values())
        self.assertEqual((len(embs), len(embs[c.FIRST_IDX])), 
            (len(self.comparison_corpus), c.EMB_SIZE))

    def test__calculate_cosine_similarity_close(self):
        """
        Similar text should have cosine similarity values 
        close to but below 1.
        """
        emb_happy_birthday = self.cosine_sim_ranker.comparison_dict.get(
                        self.comparison_corpus[c.FIRST_IDX])
        emb_party_invitations = self.cosine_sim_ranker.comparison_dict.get(
                        self.comparison_corpus[c.SECOND_IDX])
        cos_sim = self.cosine_sim_ranker._calculate_cosine_similarity(
                                emb_happy_birthday, emb_party_invitations)
        self.assertGreater(cos_sim, c.COSINE_SIMILAR_THRESH)
        self.assertLess(cos_sim, c.COSINE_SAME_VAL)

    def test__calculate_cosine_similarity_same(self):
        """The same text should have cosine similarity 1."""
        emb = self.cosine_sim_ranker.comparison_dict.get(
                        self.comparison_corpus[c.FIRST_IDX])
        cos_sim = self.cosine_sim_ranker._calculate_cosine_similarity(
                                                            emb, emb)
        self.assertEqual(cos_sim, c.COSINE_SAME_VAL)

    def test_rank_on_similarity(self):
        """Results should be ordered by similarity to the target."""
        target = 'birthday presents'
        ranked = self.cosine_sim_ranker.rank_on_similarity(target)

        #closest is "happy birthday"
        self.assertEqual(
            ranked[c.FIRST_IDX][c.FIRST_IDX], 
            self.comparison_corpus[c.FIRST_IDX]
            )

        #farthest is "wedding party"
        self.assertEqual(
            ranked[-1][c.FIRST_IDX], 
            self.comparison_corpus[-1]
            )

        #cosine similarity should be greater for more similar text
        self.assertGreater(
            ranked[c.FIRST_IDX][c.SECOND_IDX], 
            ranked[-1][c.SECOND_IDX]
            )

if __name__ == '__main__':
    unittest.main()