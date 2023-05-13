import numpy as np
from mlr import MLR
from colorama import Fore


def HHO(n, m, N, data, MLR, max_iter):
    # initialize positions of hawks
    X = np.zeros((N, n))
    for i in range(N):
        X[i] = np.random.randint(2, size=n)

    # initialize position of eagle
    E = np.zeros(n)
    E_score = float("inf")

    # main loop
    for t in range(max_iter):
        # update fitness values of hawks
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = MLR(X[i], data)

        # find the best three hawks
        sorted_fitness_idx = np.argsort(fitness)
        best1 = X[sorted_fitness_idx[0]]
        best2 = X[sorted_fitness_idx[1]]
        best3 = X[sorted_fitness_idx[2]]

        # update eagle's position
        E = (best1 + best2 + best3) / 3
        E_score = MLR(E, data)

        # update positions of hawks
        a = 2 - t * ((2) / max_iter)  # decrease a linearly from 2 to 0

        for i in range(N):
            r1 = np.random.random(n)
            r2 = np.random.random(n)
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * best1 - X[i])
            X1 = best1 - A * D
            r1 = np.random.random(n)
            r2 = np.random.random(n)
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * best2 - X[i])
            X2 = best2 - A * D
            r1 = np.random.random(n)
            r2 = np.random.random(n)
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * best3 - X[i])
            X3 = best3 - A * D
            X[i] = (X1 + X2 + X3) / 3

            # check boundaries
            X[i] = np.clip(X[i], 0, 1)
            # check boundaries
            X[i] = np.clip(X[i], 0, 1)
        # print best score at each iteration
        print(Fore.GREEN + "Iteration {}: Best Score = {:.4f}".format(t + 1, E_score))

    # return best eagle's position and score
    return E


data = np.loadtxt("./data.txt", delimiter="\t")
n = 3331
N = 10
m = 5
max_iter = 3

e = HHO(n, m, N, data, MLR, max_iter)
print(e)
print(e.shape)
