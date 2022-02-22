from textsimilarity import text_models
from textsimilarity import rankers

if __name__ == '__main__':
    bert_model = text_models.BertBaseModel()
    comparison_corpus = ['happy birthday', 'party invitations', 'special menu']
    cosine_sim_ranker = rankers.CosineSimilarityRanker(bert_model, comparison_corpus)
    print(cosine_sim_ranker.rank_on_similarity('birthday presents'))