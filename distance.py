from math import sqrt


def distance(x, y, sq=True):

    if len(x) == len(y):  # Checks that vectors have the same number of featuyres (or dimension)
        s = 0
        for i in range(0, len(x)):
            s = s + (x[i] - y[i])**2
        if sq:  # sq = True (default: True) returns the Euclidean distance, otherwise returns the squared distance
            dist = sqrt(s)
        else:
            dist = s

        return dist

    else:
        print('Vectors must have the same dimension; actual dimensions are: len(x) = ' + str(len(x)) + ', len(y) = ' + str(len(y)) + '.')

        return None
