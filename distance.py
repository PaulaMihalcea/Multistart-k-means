from math import sqrt


def distance(x, y):

    if len(x) == len(y):  # Controlla che i punti abbiano la stessa dimensione, altrimenti la distanza non può essere calcolata
        s = 0
        for i in range(0, len(x)):
            s = s + (x[i] - y[i])**2
        dist = sqrt(s)

        return dist

    else:
        print('Vectors must have the same dimension; actual dimensions are: len(x) = ' + str(len(x)) + ', len(y) = ' + str(len(y)) + '.')

        return None
