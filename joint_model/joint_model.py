import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from entropy.entropy_transfer import EntropyTransfer
from langauge_model.language_model import LanguageModel
from topic_model.lda_model import LDAModel
from topic_model.topic_model import TopicModel
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

def sample_linear(candidates, probs):
    prob = random.random()
#    random.shuffle(candidates)
    for i, word in enumerate(candidates):
        prob -= probs[i]
        if prob <= 0:
            return word
    return word

def term_wwl(word, topic):
    
    if word.split('.')[1] == '':
        return -1
    try:
        word_synset = wn.synset(word + '.01')
    except WordNetError:
        return -1
#    print(word_synset)
    topic_id = random.randrange(topic.lda_model.num_topics)
    topic_set = topic.lda_model.show_topic(topic_id)
    
#    print(topic_set)
    
    distances = []
    topic_probs = []
    for term, topic_prob in topic_set:
        if term.split('.')[1] == '':
            distances.append(-1)
        else:
            try:
                sim = word_synset.path_similarity(wn.synset(term + '.01'))
                if sim is None:
                    distances.append(0)
                else:
                    distances.append(sim)
            except WordNetError:
                distances.append(-1)
        topic_probs.append(topic_prob)
    
#    print(distances)
    tot_dist = sum(list(filter(lambda d: d>=0, distances)))
    distances = [tot_dist/len(distances) if d < 0 else d for d in distances]
    
#    distances = [word_synset.path_similarity(wn.synset(term[0] + '.01')) for term in topic_set]
    max_index = distances.index(max(distances))
    closest_term, max_prox = topic_set[max_index][0], distances[max_index]
    return topic_probs[max_index]*(max_prox + 1)

def doc_wwl(doc, topic_model):
    likelihood = 1
#    bow = topic_model.lda_model[topic_model.dictionary.doc2bow(doc)]
    for word in doc:
        likelihood *= term_wwl(word, topic_model)
    return likelihood

def print_sentence(word_sequence):
    for word in word_sequence:
        lemma = word.split('.')[0]
        print(lemma + ' ', end='')
    print()

language_power = 1

def generate_joint(doc_length, language_model, topic_model, joint_model):
    
    # Generate new document
    gen_doc = []
    
    wwl_values = dict()
    for _, word in joint_model.bow:
        word_wwl = term_wwl(word, topic_model)
        wwl_values[word] = word_wwl
    
    wwl = 1
    for i in range(doc_length):
        # Greedily choose word with minimum joint distance
        word_distances = []
        words = []
        words.clear()
        word_distances.clear()
        for _, word in joint_model.bow:
            if word == "END.":
                continue
            word_wwl = wwl_values[word]
            topic_distance = wwl * word_wwl
#            language_likelihood = language_model.smooth_prob(word)
            # N-gram
            if len(gen_doc) == 0:
                language_likelihood = language_model.smooth_prob(("BEGIN", word))
            else:
                language_likelihood = language_model.smooth_prob((gen_doc[-1], word))
            joint_dist = topic_distance*math.pow(language_likelihood, 1/language_power)
            word_distances.append(joint_dist)
            words.append(word)
                
        tot_dist = sum(list(filter(lambda d: d>=0, word_distances)))
        word_distances = [tot_dist/len(word_distances) if d < 0 else d for d in word_distances]
        total_distance = sum(word_distances)
        word_probs = [d/total_distance for d in word_distances]
        sampled_word = sample_linear(words, word_probs)
        
#        best_word = word_distances.index(max(word_distances))
        wwl *= wwl_values[sampled_word]
        gen_doc.append(sampled_word)
        
    return gen_doc

def main():
    target_file = "data/trumpTweets.csv"
#    test_file = "data/test_file_right.txt"
    test_file = "data/KimKardashianTweets.csv"
    style_docs = parse(target_file, "tweets_alt")
    
    content_docs = parse(test_file, "tweets2_alt")

    entropy_transfer_style = EntropyTransfer(0.1, 0.1, 0.2, 0.5)
    entropy_transfer_style.learn_target(style_docs[:4000])
#    target_lm = entropy_transfer_style.word_lm
    target_lm = entropy_transfer_style.bigram_lm
    
    entropy_transfer_content = EntropyTransfer(0.1, 0.1, 0.2, 0.5)
    entropy_transfer_content.learn_target(content_docs[:4000])
    
    entropy_transfer_joint = EntropyTransfer(0.1, 0.1, 0.2, 0.5)
    entropy_transfer_joint.learn_target(style_docs[:4000] + content_docs[:4000])
    
    joint_lm = entropy_transfer_joint.bigram_lm
    
    print("Training LDA...")
#    target_tm = TopicModel(entropy_transfer_content.bows)
#    lsi = target_tm.lsi_model
    target_tm = LDAModel(entropy_transfer_content.bows)
    lda = target_tm.lda_model
    print("LDA done")
    
#    print(lda.show_topics())
    print("----------")

    joint_results = []
    
#    test_docs = parse(test_file)
    for doc in content_docs[:20]:
#        tokens = entropy_transfer_content.tokenizer.tokenize(doc)
        modified_sentence = generate_joint(15, target_lm, target_tm, joint_lm)
        print_sentence(modified_sentence)
        print("----------")
        (log_likelihood, similarity) = calc_stats(target_lm, target_tm, doc, modified_sentence)
        joint_results.append(modified_sentence)


if __name__ == "__main__":
    main()
