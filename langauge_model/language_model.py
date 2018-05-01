"""
This module creates a language model from a bag of words representation
It provides means of returning smoothed probabilities, sampling,
and calculating log likelihoods.

Supports n-gram language models models by passing in an n-1 LM as a reference
"""

import math
import random


class LanguageModel:

    def __init__(self, n, bow, delta, reference_lm=None):

        self.vocabulary_size = len(bow.keys())
        self.delta = delta
        self.n = n
        self.reference_lm = reference_lm

        self.total_counts = 0
        for ngram in bow.keys():
            self.total_counts += bow[ngram]

        self.bow = bow

    def get_count_safe(self, ngram):
        if ngram in self.bow.keys():
            return self.bow[ngram]
        else:
            return 0

    def smooth_prob(self, ngram):
        count = self.get_count_safe(ngram)
        if self.n == 1:
            return (count + self.delta) / (self.total_counts + self.delta * self.vocabulary_size)
        else:
            # TODO: better smoothing for n-grams
            return (count + self.delta) / (self.delta * self.vocabulary_size) + self.reference_lm.get_count_safe(ngram[:self.n-1])

    def log_prob(self, sequence):
        log_likelihood = 0
        for ngram in sequence:
            log_likelihood += math.log(self.smooth_prob(ngram))
        return log_likelihood

    def sample(self, context=None):
        if self.n == 1:
            return self.sample_uni()[0]
        else:
            prob = random.random()
            # select the last (n-1) terms of the context
            n_minus_1 = self.n - 1
            n_minus_1_gram = tuple(context[-n_minus_1:])
            # analyze ngrams that start with that tuple
            candidates = list(filter(lambda key: key[:self.n - 1] == n_minus_1_gram, self.bow.keys()))
            # if there are none or none with sufficient probability, return a random one
            if len(candidates) == 0:
                random_index = int(random.random()*self.vocabulary_size)
                return list(self.bow.keys())[random_index][-1]
            for ngram in candidates:
                random.shuffle(candidates)
                if context == [] or ngram[:self.n-1] == n_minus_1_gram:
                    prob -= self.smooth_prob(ngram)
                    if prob <= 0:
                        return ngram[-1:][0]
            random_index = int(random.random() * self.vocabulary_size)
            return list(self.bow.keys())[random_index][-1]

    def sample_uni(self):
        prob = random.random()
        candidates = list(self.bow.keys())
        random.shuffle(candidates)
        for word in candidates:
            prob -= self.smooth_prob(word)
            if prob <= 0:
                return word
