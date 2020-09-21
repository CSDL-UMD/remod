def euc_distance(vec1, vec2):
    # euclidean
    total = 0
    for idx, dim in vec1:
        diff = vec2[idx] - vec1[idx]
        total += diff * diff
    return sqrt(total)

filepath = '../examples/test.emd'

with open(filepath, 'r') as f:
    