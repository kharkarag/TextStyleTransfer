import math

# Takes a vector (list of tuples) and returns l2 norm of its values
def vector_length(vector):
    return math.sqrt(sum(math.pow(v[1], 2) for v in vector))


def cosine_similarity(vector1, vector2):
    score = 0
    for (val1, val2) in zip(vector1, vector2):
        score += val1[1] * val2[1]
    norm_factor = vector_length(vector1) * vector_length(vector2)
    return score / norm_factor


def k_folds(documents, k):
    fold_size = int(len(documents) / k)
    # folds = [documents[i * fold_size:i * fold_size + fold_size] for i in range(k)]
    train_test_folds = []
    for test_fold in range(k):
        train_docs = []
        test_docs = []
        for i in range(k):
            fold_start = i * fold_size
            if i != test_fold:
                train_docs = train_docs + documents[fold_start:fold_start + fold_size]
            else:
                test_docs = documents[fold_start:fold_start + fold_size]
        train_test_folds.append( (train_docs, test_docs))

    return train_test_folds
