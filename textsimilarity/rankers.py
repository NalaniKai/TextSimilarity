"""
This module contains classes for rankers which
take in a model to use and corpus to compare.

Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import torch.nn.functional as F
from textsimilarity import constants as c


class CosineSimilarityRanker():
    """
    Stores text phrases and their embeddings to rank text
    based on cosine similarity.
    """

    def __init__(self, model, comparison_corpus):
        """
        model: language model
        comparison_corpus: list of strings for ranking

        Create dictionary of text phrases for comparison with
        the corresponding embeddings using the given model.
        """
        self.model = model
        self.comparison_dict = self._get_embeddings_dict(comparison_corpus)

    def _get_embeddings_dict(self, comparison_corpus):
        """
        comparison_corpus: list of strings for ranking

        Return a dictionary mapping original text
        phrases to their respective embeddings.
        """
        temp_comparison_dict = dict()
        for text in comparison_corpus:
            embedding = self.model._get_embedding(text)
            temp_comparison_dict.update({text: embedding})
        return temp_comparison_dict

    def _calculate_cosine_similarity(self, emb0, emb1):
        """
        emb0, emb1: vector embeddings from the model

        Calculate and return cosine similarity.
        """
        return F.cosine_similarity(emb0, emb1, dim=0).item()

    def rank_on_similarity(self, target_text):
        """
        target_text: text string to rank the corpus on

        Returns ordered text phrases based on cosine similarity to
        a given target text string.
        """
        target_emb = self.model._get_embedding(target_text)
        cosine_sim_tracker = []
        for text, emb in self.comparison_dict.items():
            cosine_sim = self._calculate_cosine_similarity(target_emb, emb)
            cosine_sim_tracker.append((text, cosine_sim))
        ranked_similarity = sorted(
                                    cosine_sim_tracker,
                                    key=lambda pair: pair[c.SECOND_IDX],
                                    reverse=True
                                )
        return ranked_similarity
