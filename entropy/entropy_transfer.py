"""
This module learns a language model and POS structure of a target
and applies transformations to a sentence to match
"""


import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from util.file_parsers import *
from langauge_model.language_model import LanguageModel
import math
import random


class EntropyTransfer:

    def __init__(self, delta_tags=0.1, delta_words=0.1, epsilon=0.2, p=0.5):
        self.delta_tags = delta_tags
        self.delta_words = delta_words
        self.epsilon = epsilon
        self.p = p

        # The bag of words for the style we want to replicate
        self.target_bow = {}
        self.target_entropy = 0

        # counts for the POS tags
        self.unigram_tag_counts = {}
        self.bigram_tag_counts = {}
        self.trigram_tag_counts = {}

        self.unigram_tag_lm = self.bigram_tag_lm = self.trigram_tag_lm = self.word_lm = None

        self.tag_word_counts = {}
        self.tag_word_lm = None

        # not all nltk tags are valid in wordnet
        self.allowed_tags = ['n', 'v', 'r', 'a']

        self.tokenizer = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()

        self.bows = []  # list of bows per doc for gensim

    def update_tag_bow(self, tags, n, bow):
        for tag in nltk.ngrams(tags, n):
            self.update_bow(tag, bow)

    def update_bow(self, key, bow):
        if key in bow.keys():
            bow[key] += 1
        else:
            bow[key] = 1

    # Takes a list of documents and returns a dictionary of word counts
    def learn_bow(self, docs):
        bow = {}
        for doc in docs:
            doc_bow = {}
            tokens = self.tokenizer.tokenize(doc)
            # skip tweets like 'photo'
            if len(tokens) <= 1:
                continue
            token_tags = [("BEGIN", "BEGIN")] + nltk.pos_tag(tokens) + [("END", "END")]
            for (token, tag) in token_tags:
                if token.startswith("http") or token.startswith("pic.twitter.com"):
                    continue
                wn_tag = self.wn_tag(tag)
                lemma = self.lemmatizer.lemmatize(token)
                if wn_tag in self.allowed_tags:
                    lemma = self.lemmatizer.lemmatize(token, wn_tag)
                word = lemma + '.' + wn_tag
                if not (lemma == "BEGIN" or lemma == "END"):
                    # update bag of words of just lemmas
                    self.update_bow(lemma, doc_bow)
                    # update bag of words of lemma.tag
                    self.update_bow(word, bow)
                # update bag of words of (tag, lemma)
                tag_word = (tag, lemma)
                self.update_bow(tag_word, self.tag_word_counts)
                # update unigram, bigram, and trigram tags
                tags = list(map(lambda t: t[1], token_tags))
                self.update_tag_bow(tags, 1, self.unigram_tag_counts)
                self.update_tag_bow(tags, 2, self.bigram_tag_counts)
                self.update_tag_bow(tags, 3, self.trigram_tag_counts)
            self.bows.append(doc_bow)

        return bow

    # Parses a file and learns language models
    def learn_target(self, docs):
        self.target_bow = self.learn_bow(docs)
        self.target_entropy = self.calc_avg_entropy(self.target_bow.keys())
        self.word_lm = LanguageModel(1, self.target_bow, self.delta_words)
        self.unigram_tag_lm = LanguageModel(1, self.unigram_tag_counts, self.delta_tags)
        self.bigram_tag_lm  = LanguageModel(2, self.bigram_tag_counts, self.delta_tags, self.unigram_tag_lm)
        self.trigram_tag_lm = LanguageModel(3, self.trigram_tag_counts, self.delta_tags, self.bigram_tag_lm)
        self.tag_word_lm = LanguageModel(2, self.tag_word_counts, self.delta_words, self.unigram_tag_lm)


    # Converts nltk TreeBank POS tags to WordNet format
    def wn_tag(self, tag):
        # TODO: dont add proper nouns?
        # adjectives start with J in treebank
        if tag.startswith('J'):
            return 'a'
        # the rest are just lowercase versions of the first letter
        elif tag[0].lower() in self.allowed_tags:
            return tag[0].lower()
        else:
            return tag

    # Explores wordnet to find synonyms of a given lemma
    def get_synonyms(self, lemma, pos):
        for synset in wn.synsets(lemma, pos):
            for syn in synset.lemma_names():
                if syn != lemma:
                    yield syn

    # Calculates the entropy of a word based on the frequencies of its synonyms
    def calc_word_entropy(self, word):
        split = word.split('.')
        if len(split) > 2:
            return 0
        (lemma, tag) = word.split('.')

        synonyms = []
        if tag in self.allowed_tags:
            synonyms = list(self.get_synonyms(lemma, tag))
        syn_counts = []
        for synonym in synonyms:
            syn_word = synonym + '.' + tag
            if syn_word in self.target_bow.keys():
                syn_counts.append(self.target_bow[syn_word])
        # add original word
        if word in self.target_bow.keys():
            syn_counts.append(self.target_bow[word])
        total = float(sum(syn_counts))

        num_syns = len(syn_counts)
        word_entropy = 0
        for count in syn_counts:
            proportion = count/total
            if num_syns > 1:
                word_entropy -= proportion * math.log(proportion, num_syns)
        return word_entropy

    # Averages entropy across a word sequence
    def calc_avg_entropy(self, word_sequence):
        total_entropy = 0
        for word in word_sequence:
            word_entropy = self.calc_word_entropy(word)
            total_entropy += word_entropy
        avg_entropy = total_entropy / float(len(word_sequence))
        return avg_entropy

    def swap_word(self, word):
        (lemma, tag) = word.split('.', 1)

        synonyms = []
        if tag in self.allowed_tags:
            synonyms = list(self.get_synonyms(lemma, tag))
        best_count = 0
        if word in self.target_bow.keys():
            best_count = self.target_bow[word]
        best_word = word

        for synonym in synonyms:
            key = synonym + '.' + tag
            if key in self.target_bow.keys():
                count = self.target_bow[key]
                if count > best_count:
                    best_count = count
                    best_word = key
        return best_word

    def preprocess(self, text_input):
        tokens = self.tokenizer.tokenize(text_input)
        tokens = list(filter(lambda t: not t.startswith("http"), tokens))
        words = []
        tags = list(map(lambda t: t[1], nltk.pos_tag(tokens)))
        for (token, tag) in zip(tokens, tags):
            wn_tag = self.wn_tag(tag)
            lemma = self.lemmatizer.lemmatize(token)
            if wn_tag in self.allowed_tags:
                lemma = self.lemmatizer.lemmatize(token, wn_tag)
            word = lemma + '.' + wn_tag
            words.append(word)
        return words

    # Given a sentence, preprocess it and apply the learned style
    def transfer_input(self, text_input):
        return self.transfer(self.preprocess(text_input))

    # Given a preprocessed word sequence, make substitutions to match the target entropy
    def transfer(self, word_sequence):
        target_entropy = self.calc_avg_entropy(self.target_bow.keys())

        new_sentence = []
        for word in word_sequence:
            word_entropy = self.calc_word_entropy(word)
            if math.fabs(word_entropy - target_entropy) > self.epsilon:
                # swap_word
                new_word = self.swap_word(word)
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        return new_sentence

    # Given a text string, tokenize and POS tag it
    # Then generate a sample POS structure based on the source
    # Align the two, then swap words using entropy
    def transfer_by_pos(self, word_sequence):
        word_sequence = self.tokenizer.tokenize(word_sequence)

        # Generate a POS sequence based on the target
        first_tag = "BEGIN"
        second_tag = self.bigram_tag_lm.sample([first_tag])
        tags = [first_tag, second_tag]
        min_length = len(word_sequence) / 2
        max_length = len(word_sequence) * 2

        while len(tags) < max_length:
            new_tag = self.trigram_tag_lm.sample(tags)
            if new_tag == "END":
                if len(tags) > min_length:
                    break
            else:
                tags.append(new_tag)

        # map the tags to a simpler, wordnet representation to allow more permissive matching
        wn_tags = list(map(lambda t: self.wn_tag(t), tags[1:]))

        # Fit words in the source sentence to the structure
        source_pos = list(map(lambda wt: (wt[0], self.wn_tag(wt[1])), nltk.pos_tag(word_sequence)))
        words = []
        for tag in wn_tags:
            source_tags = list(map(lambda t: t[1], source_pos))
            if tag in source_tags:
                index = source_tags.index(tag)
                words.append(source_pos[index][0])
                source_pos.pop(index)
            else:
                # random chance to sample from target or ignore tag
                # Don't sample nouns because that is highly tied to content
                prob = random.random()
                if tag != 'n' and prob < self.p:
                    word = self.tag_word_lm.sample([tag])  # sample from p(w|t)
                    words.append(word)

        # Then transfer entropy
        return self.transfer(self.preprocess(" ".join(words)))

    @staticmethod
    def print_sentence(word_sequence):
        for word in word_sequence:
            lemma = word.split('.')[0]
            print(lemma + ' ', end='')
        print()

