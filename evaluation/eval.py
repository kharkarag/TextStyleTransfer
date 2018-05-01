"""
This module is a framework for evaluating how well style has been transferred
This is done by constructing a language model and comparing the log likelihoods
of the original and transferred style sentence.
"""

from entropy.entropy_transfer import EntropyTransfer
from topic_model.topic_model import TopicModel
import matplotlib.pyplot as plt
import math


def lm_similarity(language_model, doc1, doc2):
    return math.fabs(language_model.log_prob(doc1) - language_model.log_prob(doc2))


def print_top_topics(topic_model, topic_mixture, n=10):
    topic_mixture.sort(key=lambda id_prob: id_prob[1])
    for (topic_id, prob) in topic_mixture[:n]:
        print("\t " + str(topic_model[topic_id]) + " \n \t with prob " + str(prob))


def calc_stats(language_model, topic_model, original_sentence, modified_sentence):
    log_likelihood = language_model.log_prob(modified_sentence)
    per_word = log_likelihood / float(len(modified_sentence))
    lemmas = [w.split('.')[0] for w in modified_sentence]  # remove POS tag for topic model
    similarity = topic_model.topic_similarity(original_sentence, " ".join(lemmas))
    return per_word, similarity


def main():
    target_file = "/Users/atallahhezbor/TextStyleTransfer/entropy/trumpTweets.csv" #input("Enter a file to learn the style of: ")
    filetype = "tweets"  # input("Enter a format of the file <tweets, lyrics>. Leave blank for raw text: ")
    test_file = "../entropy/test_file_right.txt" #input("Enter a file to transfer style to: ")

    if len(filetype) == 0:
        filetype = None

    # learn target language model and entropy
    entropy_transfer = EntropyTransfer()
    entropy_transfer.learn_target(target_file, filetype)
    target_lm = entropy_transfer.word_lm

    # learn target topic model
    target_tm = TopicModel(entropy_transfer.bows)
    lsi = target_tm.lsi_model
    target_topics = lsi.show_topics()

    # in the form (original text, modified text, original log likelihood, new log likelihood, topic similarity)
    entropy_results = []
    sentence_structure_results = []

    test_file_stream = open(test_file)
    # Assumes each line is a sentence. TODO: generalize
    for source_sentence in test_file_stream:
        sequence = entropy_transfer.preprocess(source_sentence)
        original_log_likelihood = target_lm.log_prob(sequence)
        per_word_ll = original_log_likelihood / float(len(sequence))

        print(f"Original Sentence - LogLikelihood:{per_word_ll}")

        # Entropy approach
        modified_sentence = entropy_transfer.transfer_input(source_sentence)
        (log_likelihood, similarity) = calc_stats(target_lm, target_tm, source_sentence, modified_sentence)
        print(f"Entropy approach - Loglikelihood:{log_likelihood} - TopicSimilarity:{similarity}")
        entropy_transfer.print_sentence(modified_sentence)
        entropy_results.append((source_sentence, modified_sentence, per_word_ll, log_likelihood, similarity))

        # Sentence structure approach
        modified_sentence = entropy_transfer.transfer_by_pos(source_sentence)
        (log_likelihood, similarity) = calc_stats(target_lm, target_tm, source_sentence, modified_sentence)
        print(f"Sentence Structure approach - Loglikelihood:{log_likelihood} - TopicSimilarity:{similarity}")
        entropy_transfer.print_sentence(modified_sentence)
        sentence_structure_results.append((source_sentence, modified_sentence, per_word_ll, log_likelihood, similarity))

    # Plot
    x = range(len(entropy_results))
    baseline_ll = list(map(lambda res: res[2], entropy_results))
    entropy_ll = list(map(lambda res: res[3], entropy_results))
    pos_ll = list(map(lambda res: res[3], sentence_structure_results))

    entropy_topic_sim = list(map(lambda res: res[4], entropy_results))
    pos_topic_sim = list(map(lambda res: res[4], sentence_structure_results))

    plt.plot(x, baseline_ll, label="Baseline Approach")
    plt.plot(x, entropy_ll, label="Entropy Approach")
    plt.plot(x, pos_ll, label="Sentence Structure Approach")
    plt.legend()
    plt.xlabel("Sentence number")
    plt.ylabel("Per-word Log likelihood")
    plt.xticks(x)
    plt.show()

    plt.plot(x, entropy_topic_sim, label="Entropy Approach")
    plt.plot(x, pos_topic_sim, label="Sentence Structure Approach")
    plt.legend()
    plt.xlabel("Sentence number")
    plt.ylabel("Topic Similarity")
    plt.xticks(x)
    plt.show()


if __name__ == "__main__":
    main()
