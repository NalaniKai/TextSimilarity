"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import nltk
from nltk.util import ngrams
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance
import spacy
import spacy.cli
from profanity_filter import ProfanityFilter
from textsimilarity import constants as c

class CleanText():
    """
    Cleans text by checking for profanity and 
    correcting spelling errors.
    """

    def __init__(self, ngram_chars=2):
        """
        Load required English vocabularies for spelling correction
        and checking for profanity.
        """
        nltk.download('words', quiet=True)
        self.english_words = words.words()
        self.ngram_chars = ngram_chars
        spacy.cli.download('en_core_web_lg')
        self.spacy_english = spacy.load('en_core_web_lg', quiet=True)
        pf = ProfanityFilter(nlps={'en': self.spacy_english})
        self.spacy_english.add_pipe(pf.spacy_component, last=True)


    def determine_text_profanity(self, text):
        """Determine whether text string contains profanity"""
        text = self.spacy_english(text)
        is_profanity = text._.is_profane
        if is_profanity:
            return True
        return False

    def spelling_correction(self, text):
        """
        Correct spelling of words in given phrase.
        Return the text string with corrections for misspelled words.
        """
        words = text.split()
        corrected = []
        for word in words:
            temp = [
                (
                    self._calculate_jaccard_distance(word, w),
                    w
                )
                for w in self.english_words 
                if word[c.FIRST_IDX] == w[c.FIRST_IDX]
            ]
            correction = sorted(temp, key=lambda v: v[c.FIRST_IDX])
            correction = correction[c.FIRST_IDX][c.SECOND_IDX]
            corrected.append(correction)
        return ' '.join(corrected)

    def _calculate_jaccard_distance(self, w0, w1):
        """Calculate and return the Jacard distance between words."""
        w0_ngrams = set(ngrams(w0, self.ngram_chars))
        w1_ngrams = set(ngrams(w1, self.ngram_chars))
        return jaccard_distance(w0_ngrams, w1_ngrams)