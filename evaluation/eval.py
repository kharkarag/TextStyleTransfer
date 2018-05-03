"""
This module is a framework for evaluating how well style has been transferred
This is done by constructing a language model and comparing the log likelihoods
of the original and transferred style sentence.
"""

from entropy.entropy_transfer import EntropyTransfer
from topic_model.topic_model import TopicModel
from util.helper_functions import *
from util.file_parsers import parse
import matplotlib.pyplot as plt
import math
import nltk
import numpy as np


def lm_similarity(language_model, doc1, doc2):
    return math.fabs(language_model.log_prob(doc1) - language_model.log_prob(doc2))


def calc_stats(language_model, topic_model, original_sentence, modified_sentence):
    log_likelihood = language_model.log_prob(modified_sentence)
    per_word = log_likelihood / float(len(modified_sentence))
    lemmas = [w.split('.')[0] for w in modified_sentence]  # remove POS tag for topic model
    similarity = topic_model.topic_similarity(original_sentence, " ".join(lemmas))
    return per_word, similarity


def average_stats(stats):
    total_similarity = 0
    total_ll = 0
    total_inc_in_ll = 0
    num_test = float(len(stats))
    max_ll = float('-inf')
    best_res = ''
    for res in stats:
        if res[3] > max_ll and len(res[1]) > 3:
            max_ll = res[3]
            best_res = (res[0], res[1])
        total_similarity += res[4]
        total_ll += res[3]
        total_inc_in_ll += res[3] - res[2]

    print(f'Best sentence was {best_res[0]} -> {best_res[1]} with ll {max_ll}')
    average_similarity = total_similarity / num_test
    average_ll = total_ll / num_test
    average_inc_ll = total_inc_in_ll / num_test

    return average_ll, average_inc_ll, average_similarity


def main():
    target_file = "data/trumpTweets.csv" #input("Enter a file to learn the style of: ")
    filetype = "tweets"  # input("Enter a format of the file <tweets, lyrics>. Leave blank for raw text: ")
    test_file = "data/test_file_right.txt" #input("Enter a file to transfer style to: ")

    if len(filetype) == 0:
        filetype = None

    docs = parse(target_file, filetype)


    # use best delta
    entropy_transfer = EntropyTransfer(0.1, 0.1, 0.2, 0.5)
    entropy_transfer.learn_target(docs[:4000])
    target_lm = entropy_transfer.word_lm

    # learn target topic model
    target_tm = TopicModel(entropy_transfer.bows)
    lsi = target_tm.lsi_model

    # in the form (original text, modified text, original log likelihood, new log likelihood, topic similarity)
    entropy_results = []
    sentence_structure_results = []

    test_docs = parse(test_file, "tweets2")[:200]

    # learn topic model of source and target
    # Used to compute topic similarities
    entropy_transfer_source = EntropyTransfer(0.1, 0.1, 0.2, 0.5)
    entropy_transfer_source.learn_target(test_docs)
    source_target_tm = TopicModel(entropy_transfer.bows + entropy_transfer_source.bows)

    num_valid_sims = 0
    for doc in test_docs:
        sequence = entropy_transfer.preprocess(doc)
        original_log_likelihood = target_lm.log_prob(sequence)
        if len(sequence) == 0:
            continue
        per_word_ll = original_log_likelihood / float(len(sequence))
        print(doc)
        print(f"Original Sentence - LogLikelihood:{per_word_ll}")

        # Entropy approach
        modified_sentence = entropy_transfer.transfer_input(doc)
        (log_likelihood, similarity) = calc_stats(target_lm, source_target_tm, doc, modified_sentence)
        if similarity is not None:
            print(f"Entropy approach - Loglikelihood:{log_likelihood} - TopicSimilarity:{similarity}")
            entropy_transfer.print_sentence(modified_sentence)
            entropy_results.append((doc, modified_sentence, per_word_ll, log_likelihood, similarity))

        # Sentence structure approach
        modified_sentence = entropy_transfer.transfer_by_pos(doc)
        (log_likelihood, similarity) = calc_stats(target_lm, source_target_tm, doc, modified_sentence)
        if similarity is not None:
            print(f"Sentence Structure approach - Loglikelihood:{log_likelihood} - TopicSimilarity:{similarity}")
            entropy_transfer.print_sentence(modified_sentence)
            sentence_structure_results.append((doc, modified_sentence, per_word_ll, log_likelihood, similarity))

        print()

    # Joint model approach here.
    # joint_results = ...

    # Average results
    average_ll_entropy, average_inc_ll_entropy, average_similarity_entropy = average_stats(entropy_results)
    average_ll_pos, average_inc_ll_pos, average_similarity_pos = average_stats(sentence_structure_results)
    # average_ll_joint, average_inc_ll_joint, average_similarity_joint = average_stats(joint_results)


    # Plot Average per-word Log Likelihood
    xaxis = ["Entropy", "POS Structure", "Joint Model"]
    log_likelihood_y = [math.exp(average_ll_entropy), math.exp(average_ll_pos), 0]
    bars = plt.bar(xaxis, log_likelihood_y)
    bars[1].set_color('orange')
    bars[2].set_color('green')
    plt.ylabel("exp(Average per-word Log Likelihood)")
    plt.xlabel("Approach")
    plt.show()

    # Plot Average Similarity
    xaxis = ["Entropy", "POS Structure", "Joint Model"]
    log_likelihood_y = [average_similarity_entropy, average_similarity_pos, 0]
    bars = plt.bar(xaxis, log_likelihood_y)
    bars[1].set_color('orange')
    bars[2].set_color('green')
    plt.ylabel("Average Topic Mixture Similarity")
    plt.xlabel("Approach")
    plt.show()

    # Plot Average Increase in Log Likelihood
    xaxis = ["Entropy", "POS Structure"]
    log_likelihood_y = [average_inc_ll_entropy, average_inc_ll_pos]
    bars = plt.bar(xaxis, log_likelihood_y)
    bars[1].set_color('orange')
    plt.ylabel("Average Change in Loglikelihood")
    plt.xlabel("Approach")
    plt.show()


if __name__ == "__main__":
    main()
