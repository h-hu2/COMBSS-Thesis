import numpy as np
import pandas as pd
import time
import sys
from scipy.linalg import solve


def generate_D(p):
    return np.diag(np.random.uniform(0.5, 5, p))


if len(sys.argv) != 3:
    print("Usage: python script.py n p")
    sys.exit(1)

n = int(sys.argv[1])
p = int(sys.argv[2])

timeArray = []
for i in range(100):
    np.random.seed(i)
    m = np.array([(1/2)**k if k <= 5 else 0 for k in range(p)])
    delta = n
    sigma = 3

    X = np.random.uniform(0, p, (n, p))
    y = X @ m+np.random.normal(0, sigma, n)

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
    x = solve(A, b)
    end_time = time.process_time()

    timeArray.append(end_time - start_time) 

avgTime = np.mean(timeArray)
sdTime = np.std(timeArray)
results = {
    "solver": "SciPy Solve",
    "n": n,
    "p": p,
    "time": avgTime,
    "sd": sdTime
}

df_results = pd.DataFrame([results])
df_results.to_csv(f"p-SciPy_Solve-n-{n}-p-{p}.csv", index=False)