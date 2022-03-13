"""
Copyright (c) 2022 NalaniKai
Released Under MIT License
"""

import unittest
from textsimilarity import clean_text


class TestCleanText(unittest.TestCase):
    """
    Test methods in CleanText for
    spelling correction, profanity checking,
    and calculating the Jaccard distance.
    """

    def setUp(self):
        """
        Define test examples for correctly and
        incorrectly spelled words.
        """
        self.text_cleaner = clean_text.CleanText(2)
        self.correct_birthday = 'birthday'
        self.incorrect_birthday = 'birhtday'
        self.correct_forest = 'forest'
        self.incorrect_forest = 'foreest'

    def tearDown(self):
        """Reset setup"""
        self.text_cleaner = None
        self.correct_birthday = None
        self.incorrect_birthday = None
        self.correct_forest = None
        self.incorrect_forest = None

    def test__calculate_jaccard_distance_difference(self):
        """Jaccard distance should be greater for less similar words"""
        jaccard_dist_less = self.text_cleaner._calculate_jaccard_distance(
                            self.incorrect_birthday, self.correct_birthday)
        jaccard_dist_more = self.text_cleaner._calculate_jaccard_distance(
                            self.correct_forest, self.correct_birthday)
        self.assertGreater(jaccard_dist_more, jaccard_dist_less)

    def test__calculate_jaccard_distance_match(self):
        """Jaccard distance should be 0.0 for the same word."""
        jaccard_dist_match = self.text_cleaner._calculate_jaccard_distance(
                            self.correct_birthday, self.correct_birthday)
        self.assertEqual(jaccard_dist_match, 0.0)

    def test_spelling_correction_all(self):
        """
        Spelling corrections should be made for text
        with multiple misspelled words.
        """
        misspelled_words = self.incorrect_birthday + ' ' + \
            self.incorrect_forest
        corrected_words = self.text_cleaner.spelling_correction(
                            misspelled_words)
        self.assertEqual(
                        corrected_words,
                        self.correct_birthday + ' ' + self.correct_forest
                        )

    def test_spelling_correction_none(self):
        """
        Spelling corrections should not be made for
        correctly spelled words.
        """
        correct_words = self.correct_birthday + ' ' + self.correct_forest
        corrected_words = self.text_cleaner.spelling_correction(
                                            correct_words)
        self.assertEqual(corrected_words, correct_words)

    def test_determine_text_profanity_false(self):
        """
        Text that does not contain profanity should
        not be marked as profane.
        """
        is_profanity = self.text_cleaner.determine_text_profanity(
                                        self.correct_birthday)
        self.assertFalse(is_profanity)

    def test_determine_text_profanity_true(self):
        """Text that contains profanity should be marked profane."""
        is_profanity = self.text_cleaner.determine_text_profanity(
                                        'go to hell')
        self.assertTrue(is_profanity)
