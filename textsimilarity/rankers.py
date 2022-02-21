"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

class CosineSimilarityRanker():
    def __init__(self, model, comparison_corpus=None):
        self.model = model
        self.comparison_corpus = comparison_corpus

    def _calculate_cosine_similarity(self):
        pass

    def rank_on_similarity(self, target_text):
        pass