import nltk
from nltk.corpus import wordnet as wn
from entropy.entropy_transfer import EntropyTransfer
from langauge_model.language_model import LanguageModel
from topic_model.lda_model import LDAModel
from evaluation.eval import calc_stats
from util.file_parsers import parse
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
import random
import math


class JointGenerator:

    def __init__(self):
        # The bag of words for the style we want to replicate
        self.target_bow = {}
        self.target_entropy = 0


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

    # Parses a file and returns a dictionary of word counts
    def learn_bow(self, filename, filetype=None):
        bow = {}
        for line in parse(filename, filetype):
            local_bow = {}
            tokens = self.tokenizer.tokenize(line)
            token_tags = [("BEGIN", "BEGIN")] + nltk.pos_tag(tokens) + [("END", "END")]
            for (token, tag) in token_tags:
                wn_tag = self.wn_tag(tag)
                lemma = self.lemmatizer.lemmatize(token)
                if wn_tag in self.allowed_tags:
                    lemma = self.lemmatizer.lemmatize(token, wn_tag)
                word = lemma + '.' + wn_tag
                self.update_bow(lemma, local_bow)
                self.update_bow(word, bow)

                tag_word = (tag, lemma)
                self.update_bow(tag_word, self.tag_word_counts)
            self.bows.append(local_bow)
            tags = list(map(lambda t: t[1], token_tags))
            self.update_tag_bow(tags, 1, self.unigram_tag_counts)
            self.update_tag_bow(tags, 2, self.bigram_tag_counts)
            self.update_tag_bow(tags, 3, self.trigram_tag_counts)

        return bow

    # Parses a file and learns language models
    def learn_target(self, filename, filetype=None):
        self.target_bow = self.learn_bow(filename, filetype)
        self.target_entropy = self.calc_avg_entropy(self.target_bow.keys())
        self.word_lm = LanguageModel(1, self.target_bow, 0.3)
        self.bigram_tag_lm  = LanguageModel(2, self.bigram_tag_counts, 0.3, self.unigram_tag_lm)
        self.trigram_tag_lm = LanguageModel(3, self.trigram_tag_counts, 0.3, self.bigram_tag_lm)
        self.tag_word_lm = LanguageModel(2, self.tag_word_counts, 0.3, self.unigram_tag_lm)


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
        (lemma, tag) = word.split('.')

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
        tokens = list(filter(lambda t: t != '.', tokens))
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

    # Given a preprocessed word sequence, generate a new document
    def generate_joint(doc_length, language_model, topic_model):
        gen_doc = []
        for i in range(doc_length):
            # Greedily choose word with minimum joint distance
            word_distances = []
            for word in language_model.bow:
    #            topic_distance = topic_model.likelihood(word)
                topic_distance = doc_wwl(gen_doc + word, topic_model)
                language_likelihood = language_model.smooth_prob(word)
                # N-gram
                # language_likelihood = language_model.smooth_prob([gen_doc[-1], word])
                joint_dist = topic_distance*language_likelihood
                word_distances.append(joint_dist)
                best_word = min(word_distances)
            gen_doc.append(best_word)
            
        return gen_doc

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
            if new_tag == "END" and len(tags) > min_length:
                break
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
                if tag != 'n' and prob < 0.5:
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


def term_wwl(word, topic):
    word_synset = wn.synset(word)
    distances = [word_synset.path_similarity(wn.synset(term)) for term in topic]
    min_index = distances.index(min(distances))
    closest_term, min_distance = topic[min_index], distances[min_index]
    return topic.word_likelihood(closest_term)/(min_distance + 1)

def doc_wwl(doc, topic):
    likelihood = 1
    for word in doc:
        likelihood *= term_wwl(word, topic)
    return likelihood

def print_sentence(word_sequence):
        for word in word_sequence:
            lemma = word.split('.')[0]
            print(lemma + ' ', end='')
        print()

def generate_joint(doc_length, language_model, topic_model):
    
    # Generate new document
    gen_doc = []
    for i in range(doc_length):
        # Greedily choose word with minimum joint distance
        word_distances = []
        for word in language_model.bow:
#            topic_distance = topic_model.likelihood(word)
            topic_distance = doc_wwl(gen_doc + word, topic_model)
            language_likelihood = language_model.smooth_prob(word)
            # N-gram
            # language_likelihood = language_model.smooth_prob([gen_doc[-1], word])
            joint_dist = topic_distance*language_likelihood
            word_distances.append(joint_dist)
            best_word = min(word_distances)
        gen_doc.append(best_word)
        
    return gen_doc

def main():
    target_file = "../data/trumpTweets.csv"
    filetype = "tweets"
    test_file = "../data/test_file_right.txt"
    docs = parse(target_file, filetype)

    entropy_transfer = EntropyTransfer()
    entropy_transfer.learn_target(docs)
    target_lm = entropy_transfer.word_lm
    
    target_tm = LDAModel(entropy_transfer.bows)
    lda = target_tm.lda_model

    joint_results = []
    
    test_docs = parse(test_file)
    for doc in test_docs:
        tokens = entropy_transfer.tokenizer.tokenize(doc)
        modified_sentence = generate_joint(len(doc), target_lm, lda)
        (log_likelihood, similarity) = calc_stats(target_lm, target_tm, doc, modified_sentence)
        print_sentence(modified_sentence)
        joint_results.append(modified_sentence)


if __name__ == "__main__":
    main()
