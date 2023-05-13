from sklearn.linear_model import LinearRegression
import numpy as np
from colorama import Fore, Style
from sklearn.metrics import r2_score

# data = np.loadtxt("./data.txt", delimiter="\t")


def MLR(x, data):
    np.random.seed(123)
    x1 = np.round(x + 1).astype(int)
    rx, cx = data.shape
    x_rid = []
    for i in x1:
        x_rid.append(data[:, i])
    x_rid = np.array(x_rid).T
    y = data[:, 0]
    ones = np.ones(55)
    gg = np.insert(x_rid, 0, ones, axis=1)
    mlr = LinearRegression()
    w = mlr.fit(gg, y)
    y_predict = mlr.predict(gg)
    # err = r2_score(y,y_predict)
    oo = np.sum((y - y_predict) ** 2)
    oo1 = np.sum((y - np.mean(y_predict)) ** 2)
    err = oo / oo1
    # print(Fore.LIGHTYELLOW_EX + f"rx is : {rx} \ncx is : {cx} \nx1 is : {x1}")
    # print(Fore.LIGHTMAGENTA_EX + "__" * 15)
    # print(Fore.BLUE + f"oo is : {oo} \noo1 is : {oo1}")
    # print(Fore.RED + f"err is : {err}")
    return err


# x = np.array([2377.2, 2945.2, 2400.7, 63.0, 2247.3])
# MLR(x, data=data)
