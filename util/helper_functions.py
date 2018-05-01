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
