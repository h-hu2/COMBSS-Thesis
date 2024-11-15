import numpy as np
import pandas as pd
import time
import sys
from scipy.sparse.linalg import cg


def generate_D(p):
    return np.diag(np.random.uniform(0.5, 5, p))


if len(sys.argv) != 3:
    print("Usage: python script.py n p")
    sys.exit(1)

n = int(sys.argv[1])
p = int(sys.argv[2])
tol = 1e-7

max_iter = 10*max(n, p)
timeArray = []
converge = 0
for i in range(100):
    np.random.seed(i)
    m = np.array([(1/2)**k if k <= 5 else 0 for k in range(p)])
    delta = n
    sigma = 3

    X = np.random.uniform(0, p, (n, p))
    y = X@m+np.random.normal(0, sigma, n)

    '''
    We wish to solve the linear regression equation XBeta = y
    We can express this in the form X'XBeta = X'y, ie. ABeta = b, with
    - A = X'X, and
    - b = X'y
    '''

    # Generate A and b
    D = generate_D(p)
    A = np.transpose(X)@X + delta*D
    b = np.transpose(X)@y

    start_time = time.process_time()
    x, info = cg(A, b, rtol=tol, maxiter=max_iter)
    end_time = time.process_time()

    if info == 0:
        converge += 1

    timeArray.append(end_time - start_time)

    convergenceRate = converge/100
    avgTime = np.mean(timeArray)
    sdTime = np.std(timeArray)

    timeArray.append(end_time - start_time) 

results = {
    "solver": "SciPy CG",
    "n": n,
    "p": p,
    "time": avgTime,
    "sd": sdTime,
    "convergenceRate": convergenceRate,
    "tol": tol
}

df_results = pd.DataFrame([results])
df_results.to_csv("p-SciPy_CG-Low.csv", index=False)
# %%
