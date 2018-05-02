"""
This module creates a topic model from a list of bag of word representations of documents.
Unlike `language_model`, which is built from scratch, this module makes use of the Gensim package
"""

from gensim import corpora, models
from util.helper_functions import *


class LDAModel:

    def __init__(self, bow_list):

        # construct corpus compatible with gensim
        # (a list of tuples of (tokenID, frequency) )
        vocabulary = [bow.keys() for bow in bow_list]
        dictionary = corpora.Dictionary(vocabulary)
        corpus = []
        for bow in bow_list:
            doc = []
            for key in bow.keys():
                doc.append((dictionary.token2id[key], bow[key]))
            corpus.append(doc)
        self.corpus = corpus
        self.tfidf = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf[corpus]
        self.dictionary = dictionary
        self.lda_model = models.LdaMulticore(corpus=self.corpus_tfidf, id2word=self.dictionary, num_topics=10, workers=3)

    def doc_likelihood(self, doc):
        return self.lda_model[doc]

    def word_likelihood(self, word):
        return self.lda_model[word]

    def topic_similarity(self, doc1, doc2):
        bow1 = self.dictionary.doc2bow(doc1.split(" "))
        topic_mixture1 = self.lda_model[bow1]

        bow2 = self.dictionary.doc2bow(doc2.split(" "))
        topic_mixture2 = self.lda_model[bow2]

        return cosine_similarity(topic_mixture1, topic_mixture2)


