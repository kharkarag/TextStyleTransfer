import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import SpaceTokenizer
from nltk.stem import WordNetLemmatizer
import math


class EntropyTransfer:

    def __init__(self):
        # The bag of words for the style we want to replicate
        self.target_bow = {}
        self.target_entropy = 0

        # not all nltk tags are valid in wordnet
        self.allowed_tags = ['n', 'v', 'r', 'a']

        self.tokenizer = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    # Parses a file (csv for now) and returns a dictionary of word counts
    def learn_bow(self, filename):
        lm = {}
        f = open(filename)
        for line in f:
            # TODO: this is dependent on a particular csv format. generalize
            tweet_data = line.split(',')
            if len(tweet_data) > 3:
                tweet_text = tweet_data[2]
                tokens = self.tokenizer.tokenize(tweet_text)
                tags = nltk.pos_tag(tokens)
                for (token, tag) in zip(tokens, tags):
                    wn_tag = self.wn_tag(tag)
                    if wn_tag in self.allowed_tags:
                        lemma = self.lemmatizer.lemmatize(token, wn_tag)
                        word = lemma + '.' + wn_tag
                        if word in lm:
                            lm[word] += 1
                        else:
                            lm[word] = 1
        return lm

    def learn_target(self, filename):
        self.target_bow = self.learn_bow(filename)
        self.target_entropy = self.calc_avg_entropy(self.target_bow.keys())


    # Converts nltk TreeBank POS tags to WordNet format
    def wn_tag(self, pos_tag):
        tag = pos_tag[1]
        # TODO: dont add proper nouns?
        # adjectives start with J in treebank
        if tag.startswith('J'):
            return 'a'
        # the rest are just lowercase versions of the first letter
        elif tag[0].lower in self.allowed_tags:
            return tag[0].lower()
        else:
            return 'n' #hacky workaround to still allow lemmatization

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

        synonyms = list(self.get_synonyms(lemma, tag))
        syn_counts = []
        for synonym in synonyms:
            syn_word = synonym + '.' + tag
            if syn_word  in self.target_bow.keys():
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

        synonyms = list(self.get_synonyms(lemma, tag))
        for synonym in synonyms:
            key = synonym + '.' + tag
            # TODO: pick one that would maximize entropy
            if key in self.target_bow.keys():
                return key
        return word

    # Given a sentence, preprocess it and apply the learned style
    def transfer_input(self, text_input):
        tokens = self.tokenizer.tokenize(text_input)
        words = []
        tags = nltk.pos_tag(tokens)
        for (token, tag) in zip(tokens, tags):
            wn_tag = self.wn_tag(tag)
            lemma = self.lemmatizer.lemmatize(token, wn_tag)
            word = lemma + '.' + wn_tag
            words.append(word)
        return self.transfer(words)

    # Given a preprocessed word sequence, make substitutions to match the target entropy
    def transfer(self, word_sequence):
        target_entropy = self.calc_avg_entropy(self.target_bow.keys())
        # TODO: tune epsillon
        epsillon = 0.2
        new_sentence = []
        print("target entropy is " + str(target_entropy))
        for word in word_sequence:
            word_entropy = self.calc_word_entropy(word)
            if math.fabs(word_entropy - target_entropy) > epsillon:
                # swap_word
                new_word = self.swap_word(word)
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        return new_sentence

    @staticmethod
    def print_sentence(word_sequence):
        for word in word_sequence:
            lemma = word.split('.')[0]
            print(lemma + ' ', end='')
        print()

def main():
    entropy_transfer = EntropyTransfer()
    train_file = "tweets.csv" #input("Enter a file to transfer style from:")
    entropy_transfer.learn_target(train_file)
    source_sentence = "That is incredible, reasonable, and moral" #input("Enter a sentence to transfer style to: ")
    print("Source sentence:")
    print(source_sentence)
    new_sentence = entropy_transfer.transfer_input(source_sentence)
    print("Resulting sentence:")
    EntropyTransfer.print_sentence(new_sentence)


if __name__ == "__main__":
    main()

