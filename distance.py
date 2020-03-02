from math import sqrt


def distance(x, y, sq=False):

    if len(x) == len(y):  # Checks that vectors have the same number of featuyres (or dimension)
        s = 0
        for i in range(0, len(x)):
            s = s + (x[i] - y[i])**2
        if sq:  # sq = True (default: False) returns the squared distance, otherwise returns the Euclidean distance
            dist = s
        else:
            dist = sqrt(s)

        return dist

    else:
        print('Vectors must have the same dimension; actual dimensions are: len(x) = ' + str(len(x)) + ', len(y) = ' + str(len(y)) + '.')

        return None
