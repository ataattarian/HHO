import numpy as np
from mlr import MLR as CostFunction
import random
import math

var_num = 6
lowerBound = 1
upperBound = 3330
N = 10
max_iter = 20
data = np.loadtxt("./data.txt", delimiter="\t")


solution = {"Position": [], "Cost": []}
Hawks = [solution.copy() for _ in range(N)]

for i in range(N):
    Hawks[i]["Position"] = np.random.uniform(lowerBound, upperBound, var_num)
    Hawks[i]["Cost"] = CostFunction(Hawks[i]["Position"], data)

sorted_data = sorted(Hawks, key=lambda x: x["Cost"])
rabbit = sorted_data[0]
bestFar = np.zeros(max_iter)

# mainLoop
for it in range(1, max_iter + 1):
    E1 = 2 * (1 - (it / max_iter))
    for i in range(N):
        E0 = 2 * random.random() - 1  # -1<E0<1
        Escaping_Energy = E1 * E0  # escaping energy of rabbit
        if abs(Escaping_Energy) >= 1:
            q = random.random()
            rand_Hawk_index = random.randint(0, N - 1)
            X_rand = Hawks[rand_Hawk_index]["Position"]
            if q < 0.5:
                Hawks[i]["Position"] = X_rand - random.random() * abs(X_rand - 2 * random.random() * Hawks[i]["Position"])
            else:
                z = [Hawk["Position"] for Hawk in Hawks]
                z = np.reshape(z, (N, var_num))
                Hawks[i]["Position"] = (rabbit["Position"] - np.mean(z, axis=0)) - random.random() * np.random.uniform(lowerBound, upperBound, size=var_num)

