"""
This module creates a topic model from a list of bag of word representations of documents.
Unlike `language_model`, which is built from scratch, this module makes use of the Gensim package
"""

from gensim import corpora, models
from util.helper_functions import *


class TopicModel:

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
        self.lsi_model = models.lsimodel.LsiModel(corpus=self.corpus_tfidf, id2word=self.dictionary, num_topics=100)

    def topic_similarity(self, doc1, doc2):
        bow1 = self.dictionary.doc2bow(doc1.split(" "))
        topic_mixture1 = self.lsi_model[bow1]

        bow2 = self.dictionary.doc2bow(doc2.split(" "))
        topic_mixture2 = self.lsi_model[bow2]

        return cosine_similarity(topic_mixture1, topic_mixture2)

