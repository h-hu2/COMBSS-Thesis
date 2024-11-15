#%%
import pandas as pd
import combss
#%%
path_train = './thesisTrain.csv'
df = pd.read_csv(path_train, sep='\t', header=None)
data = df.to_numpy()
y_train = data[:, 0]
X_train = data[:, 1:]

path_test= './thesisTest.csv'
df = pd.read_csv(path_test, sep='\t', header=None)
data = df.to_numpy()
y_test = data[:, 0]
X_test = data[:, 1:]

combssOptimiser = combss.linear.model()
combssOptimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q = 8, nlam = 20, scaling=True)

# %%
combssOptimiser.subset
# %%
combssOptimiser.mse
# %%
combssOptimiser.coef_
# %%
combssOptimiser.lambda_
# %%
combssOptimiser.run_time
# %%

combssOptimiser.lambda_list

# %%

combssOptimiser.subset_list
#%%
combssOptimiser.fit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, q = 10, nlam = 50, scaling=True, tau = 0.9, delta_frac = 20)

# %%
combssOptimiser.subset
# %%
combssOptimiser.mse
# %%
combssOptimiser.coef_
# %%
combssOptimiser.lambda_
# %%
combssOptimiser.run_time
# %%
combssOptimiser.lambda_list
# %%
combssOptimiser.subset_list
