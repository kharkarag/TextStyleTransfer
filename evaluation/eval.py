"""
This module is a framework for evaluating how well style has been transferred
This is done by constructing a language model and comparing the log likelihoods
of the original and transferred style sentence.
"""

from entropy.entropy_transfer import EntropyTransfer


def main():
    target_file = input("Enter a file to learn the style of: ")
    filetype = input("Enter a format of the file <tweets, lyrics>. Leave blank for raw text: ")

    if len(filetype) == 0:
        filetype = None

    # learn target language model and entropy
    entropy_transfer = EntropyTransfer()
    entropy_transfer.learn_target(target_file, filetype)

    target_lm = entropy_transfer.word_lm
    while True:
        source_sentence = input("Enter a sentence to transfer style to: ")
        print("Source sentence:")
        print(source_sentence)
        sequence = entropy_transfer.preprocess(source_sentence)
        print("has log likelihood " + str(target_lm.log_prob(sequence)))

        # Entropy approach
        print("Resulting sentence w/ just entropy changes:")
        new_sentence = entropy_transfer.transfer_input(source_sentence)
        EntropyTransfer.print_sentence(new_sentence)
        print("has log likelihood " + str(target_lm.log_prob(new_sentence)))

        # Sentence structure approach
        print("Resulting sentence w/ sentence structure changes:")
        new_sentence = entropy_transfer.transfer_by_pos(source_sentence)
        EntropyTransfer.print_sentence(new_sentence)
        print("has log likelihood " + str(target_lm.log_prob(new_sentence)))


    # TODO: Formalize evaluation and collect and plot results


if __name__ == "__main__":
    main()
