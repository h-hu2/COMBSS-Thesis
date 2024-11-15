import numpy as np
import pandas as pd
from numpy.linalg import pinv, norm
from scipy.sparse.linalg import cg
import time
import helpers
from scipy.sparse.linalg import cg, LinearOperator


'''
COMBSS Variant 2 Functions
'''


def gamma_cg(t, X, y, Xy, delta_frac, gamma,  upsilon, g1, g2, 
			  cg_maxiter=None, 
			  cg_tol=1e-5):
	""" 
	Calculates the gradient of the objective function with respect to parameters t, as well as the 
	corresponding estimate of beta. Also returns components of the objective function used for recurrent 
	calls of this function.

	Parameters
	----------
	t : array-like of floats.
		The t vector used for calculations.

	X : array-like of shape (n_samples, n_covariates)
		The design matrix, where `n_samples` is the number of samples observed
		and `n_covariates` is the number of covariates measured in each sample.

	y : array-like of shape (n_samples)
		The response data, where `n_samples` is the number of response elements.

	Xy : array-like of shape (n_covariates, 1).
		The matrix Xy is defined as (X.T@y)/n, as featured in the original COMBSS paper.

	lam : float
		The penalty parameter used within the objective function. Referred to as
		'lambda' in the original COMBSS paper.

	delta_frac : float
		The value of delta/n, where delta is a tuning parameter as referenced in the original COMBSS paper. 

	gamma : array-like of floats of shape (n_covariates, 1)
		The current values of beta times t, calculated from the X matrix, y vector and current 
		values of vector t.

	upsilon : array-like of floats of shape (n_covariates, 1).
		The associated value of upsilon with respect to the existing ts.
	
	g1 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of beta when presented with high 
		dimensional data. In particular, it is a byproduct of the implementation of the 
		Woodbury matrix in the original COMBSS paper, section 6.1.

	g2 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of the gradient of the objective 
		function with respect to t when presented with high dimensional data. In particular, 
		it is a byproduct of the implementation of the Woodbury matrix in the original 
		COMBSS paper, section 6.1.

	cg_maxiter (Conjugate gradient parameter) : int
		The maximum number of iterations for the conjugate gradient algortihm used 
		to approximate the gradient of the function with respect to t and the gradient 
		of the objective function with respect to beta before the conjugate gradient 
		algorithm terminates.
		Default value = None.

	cg_tol (Conjugate gradient parameter) : float
		The acceptable tolerance used for the termination condition in the conjugate gradient 
		algortihms used to approximate the gradient of the function with respect to t and the 
		gradient of the objective function with respect to beta.
		Default value = 1e5.

	
	Returns
	-------
	grad : array-like of floats (n_covariates, 1).
		The derivative of the objective function with respect to t.

	upsilon : array-like of floats of shape (n_covariates, 1).
		The associated value of upsilon with respect to the existing ts.

	g1 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of beta when presented with high 
		dimensional data. In particular, it is a byproduct of the implementation of the 
		Woodbury matrix in the original COMBSS paper, section 6.1.

	g2 : array-like of floats of shape (n_samples, 1)
		The vector g1 is used in constructing the estimate of the gradient of the objective 
		function with respect to t when presented with high dimensional data. In particular, 
		it is a byproduct of the implementation of the Woodbury matrix in the original 
		COMBSS paper, section 6.1.
	"""
	
	p = t.shape[0]
	n = y.shape[0]
	
	if cg_maxiter == None:
		cg_maxiter = min(n, p)
   
	if n >= p:
		t_sqr = t*t
		t_sqr[t_sqr < 1e-8] = 1e-8  #For numerical stability, small values are mapped to 1e-8. This is not required if truncation is used.
		dia = delta_frac*(1 - t_sqr)/t_sqr    

		## Construct Mt
		def matvec(v):
			Mv = X.T@(X@v)/n + dia*v 
			return Mv
		
		M = LinearOperator((p, p), matvec=matvec)
		
		
		## Obtaining gamma estimate
		gamma, _ = cg(M, Xy, x0=gamma, maxiter=cg_maxiter, tol=cg_tol)
		
		b = X.T@(X@gamma)/n 
		upsilon, _ = cg(M, b, x0=upsilon, maxiter=cg_maxiter, tol=cg_tol) 
		
	else:
		## constructing Lt_tilde
		t_sqr = t*t
		temp = 1 - t_sqr
		temp[temp < 1e-8] = 1e-8   #For numerical stability
		inv_dia = (1/delta_frac)*t_sqr/temp    
		
		D_invXy = inv_dia*Xy
		XD_invXy = X@D_invXy
		
		def matvec(v):
			Mv = v + X@(inv_dia*(X.T@v))/n
			return Mv   
		
		M = LinearOperator((n, n), matvec=matvec)
		
		g1, _ = cg(M, XD_invXy, x0=g1, maxiter=cg_maxiter, tol=cg_tol)
		
	   
		## estimate gamma
		gamma = D_invXy - inv_dia*(X.T@g1)/n
		
		b = X.T@(X@gamma)/n 
		D_invb = inv_dia*b
		XD_invb = X@D_invb
		g2, _ = cg(M, XD_invb, x0=g2, maxiter=cg_maxiter, tol=cg_tol)
		upsilon = D_invb - inv_dia*(X.T@g2)/n

	return gamma, upsilon, g1, g2


def adam_w(X, y, beta, w0, lam, gam1 = 0.9, gam2 = 0.999, alpha = 0.1, epsilon = 10e-8, maxiter = 1e5, tol = 1e-5):

	# Initialising data-related variables
	(n, p) = X.shape 
	delta = n
	
	# Initialising Adam-related variables
	i = 0
	v_w, u_w = np.zeros(p), np.zeros(p)
	
	stop = False
	converge = False
	
	w_new = w0.copy()
	t_new = helpers.w_to_t(w_new)

	while not stop:
		
		# Initialisation parameters
		w_curr = w_new.copy()
		t_curr = helpers.w_to_t(w_curr)

		# Perform updates for w
		gradw = helpers.grad_v1_w(X, t_curr, beta, delta, lam, y, w_curr)
		v_w = gam1*v_w + (1 - gam1)*gradw
		u_w = gam2*u_w + (1 - gam2)*(gradw*gradw)
		v_ws = v_w/(1-gam1**(i+1))
		u_ws = u_w/(1-gam2**(i+1))
		w_new = w_curr - alpha*np.divide(v_ws, (np.sqrt(u_ws) + epsilon))
		t_new = helpers.w_to_t(w_new)

		# Assess stopping conditions
		if (i > maxiter):
			stop = True
		else:
			diff_t = np.linalg.norm((t_new- t_curr), 2)
			if (diff_t < tol):
				gradt_new = helpers.grad_v1_t(X, t_new, beta, delta, y)
				gradt_curr = helpers.grad_v1_t(X, t_curr, beta, delta, y)

				diff_gradt = np.linalg.norm((gradt_new - gradt_curr),2)
				if (diff_gradt < tol):
					stop = True
		# Iterate through counter
		i = i + 1
	
	if i + 1 < maxiter:
		converge = True
	
	return w_new, t_new, converge, i+1

def control_funV2(X, y, lam, t_init,
		delta_frac = 1,
		CG = True,
	 
		# Stopping criteria
		max_norm = True, # default we use max norm as the termination condition.
		epoch=5,
		
		## Truncation parameters
		tau = 0.5,
		eta = 0.0, 
		
		## Parameters for Conjugate Gradient method
		cg_maxiter = None,
		cg_tol = 1e-5,
		
		# Parameters for Adam 
		adam_maxiter = None,
		adam_tol = 1e-5,
		
		# Parameters for Gradient Descent 
		gd_maxiter = None,
		gd_tol = 1e-5):
	"""
	Implementation of the ADAM optimizer for combss. 
	"""    
	(n, p) = X.shape
	
	if cg_maxiter == None:
		cg_maxiter = n
		
	if adam_maxiter == None:
		adam_maxiter = 1000
	
	if gd_maxiter == None:
		gd_maxiter = 1000

	## One time operation
	Xy = (X.T@y)/n

	## Initialization
	t = t_init.copy()
		
	w = helpers.t_to_w(t)
	
	t_trun = t.copy()
	t_prev = t.copy()
	active = p
	
	gamma_trun = np.zeros(p)
	beta_trun = np.zeros(p)  

	upsilon = np.zeros(p)
	g1 = np.zeros(n)
	g2 = np.zeros(n)
	
	count_to_term = 0
	
	
	for l in range(gd_maxiter):
		M = np.nonzero(t)[0] ## Indices of t correponds to elements greater than eta. 
		M_trun = np.nonzero(t_trun)[0] 
		active_new = M_trun.shape[0]
		
		if active_new != active:
			X = X[:, M_trun]
			Xy = Xy[M_trun]
			active = active_new
			t_trun = t_trun[M_trun]
		
		## Compute gradient for the effective terms
		gamma_trun, upsilon, g1, g2 = gamma_cg(t_trun, X, y, Xy, delta_frac, gamma_trun[M_trun],  upsilon[M_trun], g1=g1, g2=g2)
		w_trun = w[M]
		beta_trun = gamma_trun/t_trun
		
		w_trun, t_trun, _, _ = adam_w(X, y, beta_trun, w_trun, lam, maxiter = adam_maxiter, tol = adam_tol)
		
		w[M] = w_trun
		t[M] = t_trun
		
		w[t <= eta] = -np.inf
		t[t <= eta] = 0.0
		
		beta = np.zeros(p)
		beta[M] = beta_trun

		t_trun = t[M] 

		if max_norm:
			norm_t = max(np.abs(t - t_prev))
			if norm_t <= gd_tol:
				count_to_term += 1
				if count_to_term >= epoch:
					break
			else:
				count_to_term = 0
				
		else:
			norm_t = norm(t)
			if norm_t == 0:
				break
			
			elif norm(t_prev - t)/norm_t <= gd_tol:
				count_to_term += 1
				if count_to_term >= epoch:
					break
			else:
				count_to_term = 0
		t_prev = t.copy()
	
	model = np.where(t > tau)[0]

	if l+1 < gd_maxiter:
		converge = True
	else:
		converge = False
	return  t, model, converge, l+1

def combss_dynamicV2(X, y, 
				   q = None,
				   nlam = None,
				   t_init= [],         # Initial t vector
				   tau=0.5,               # tau parameter
				   delta_frac=1, # delta_frac = n/delta
				   fstage_frac = 0.5,    #fraction lambda values explored in first stage of dynamic grid
				   eta=0.0,               # Truncation parameter
				   epoch=10,           # Epoch for termination 
				   gd_maxiter=1000, # Maximum number of iterations allowed by GD
				   gd_tol=1e-5,         # Tolerance of GD
				   cg_maxiter=None, # Maximum number of iterations allowed by CG
				   cg_tol=1e-5,     # Tolerance of CG
				   adam_maxiter=None, # Maximum number of iterations allowed by Adam
				   adam_tol = 10e-5): # Tolerance of Adam       
	"""
	Dynamic grid of lambda is generated as follows: We are given maximum model size $q$ of interest. 
	
	First pass: We start with $\lambda = \lambda_{\max} = \mathbf{y}^\top \mathbf{y}/n$, 
				where an empty model is selected, and use $\lambda \leftarrow \lambda/2$ 
				until we find model of size larger than $q$. 
	
	Second pass: Then, suppose $\lambda_{grid}$ is (sorted) vector of $\lambda$ valued exploited in 
				 the first pass, we move from the smallest value to the large value on this grid, 
				 and run COMBSS at $\lambda = (\lambda_{grid}[k] + \lambda_{grid}[k+1])/2$ if $\lambda_{grid}[k]$ 
				 and $\lambda_{grid}[k+1]$ produced models with different sizes. 
				 We repeat this until the size of $\lambda_{grid}$ is larger than a fixed number $nlam$.
	"""
	(n, p) = X.shape
	
	# If q is not given, take q = n.
	if q == None:
		q = min(n, p)
	
	# If number of lambda is not given, take it to be n.
	if nlam == None:
		nlam == n
	t_init = np.array(t_init) 
	if t_init.shape[0] == 0:
		t_init = np.ones(p)*0.5
	
	if cg_maxiter == None:
		cg_maxiter = n
	
	lam_max = y@y/n # max value for lambda

	# Lists to store the findings
	model_list = []
	
	lam_list = []
	lam_vs_size = []

	lam = lam_max
	count_lam = 0

	## First pass on the dynamic lambda grid
	stop = False
	while not stop:
		_, model, _, _ = control_funV2(X, y, lam, t_init = t_init, gd_maxiter = gd_maxiter, gd_tol = gd_tol, cg_maxiter=cg_maxiter, adam_maxiter=adam_maxiter)

		len_model = model.shape[0]

		lam_list.append(lam)
		model_list.append(model)
		lam_vs_size.append(np.array((lam, len_model)))
		count_lam += 1
		print(len_model)
		if len_model >= q or count_lam > nlam*fstage_frac:
			stop = True
		lam = lam/2
		
	stop = False
	while not stop:
		temp = np.array(lam_vs_size)
		order = np.argsort(temp[:, 1])
		lam_vs_size_ordered = np.flip(temp[order], axis=0)        

		for i in range(order.shape[0]-1):

			if count_lam <= nlam and lam_vs_size_ordered[i+1][1] <= q and  (lam_vs_size_ordered[i+1][1] != lam_vs_size_ordered[i][1]):

				lam = (lam_vs_size_ordered[i][0] + lam_vs_size_ordered[i+1][0])/2

				_, model, _, _ = control_funV2(X, y, lam, t_init = t_init, gd_maxiter = gd_maxiter, gd_tol = gd_tol, cg_maxiter=cg_maxiter, adam_maxiter=adam_maxiter)

				len_model = model.shape[0]

				lam_list.append(lam)
				model_list.append(model)
				lam_vs_size.append(np.array((lam, len_model)))    
				count_lam += 1

			if count_lam > nlam:
				stop = True
				break

	temp = np.array(lam_vs_size)
	order = np.argsort(temp[:, 1])
	model_list = [model_list[i] for i in order]
	lam_list = [lam_list[i] for i in order]
	
	return  (model_list, lam_list)

def combssV2(X_train, y_train, X_test, y_test, 
			q = None,           # maximum model size
			nlam = 50,        # number of values in the lambda grid
			t_init= [],         # Initial t vector
			tau=0.5,               # tau parameter
			delta_frac=1, # delta_frac = n/delta
			eta=0.001,               # Truncation parameter
			epoch=10,           # Epoch for termination 
			gd_maxiter=1000, # Maximum number of iterations allowed by GD
			gd_tol=1e-5,         # Tolerance of GD
			cg_maxiter=None, # Maximum number of iterations allowed by CG
			cg_tol=1e-5,     # Tolerance of CG
			adam_maxiter=None, # Maximum number of iterations allowed by Adam
			adam_tol = 10e-5): # Tolerance of Adam  
	""" 
	COMBSSV1 with SubsetMapV1
	
	This is the first version of COMBSS available in the paper. 
	In particular, we only look at the final t obtained by 
	the gradient descent algorithm (ADAM Optimizer) and consider the model corresponds 
	to significant elements of t.
	"""

	# Data Normalisation
	column_norms = np.linalg.norm(X_train, axis=0)
	X_train_hat = X_train / column_norms
	
	# Call COMBSS_dynamic with ADAM optimizer
	(n, p) = X_train.shape
	t_init = np.array(t_init) 
	if t_init.shape[0] == 0:
		t_init = np.ones(p)*0.5
		
	# If q is not given, take q = n
	if q == None:
		q = min(n, p)
	
	tic = time.process_time()
	(model_list, lam_list) = combss_dynamicV2(X_train_hat, y_train, q = q, nlam = nlam, t_init=t_init, tau=tau, delta_frac=delta_frac, eta=eta, epoch=epoch, gd_maxiter= gd_maxiter, gd_tol=gd_tol, cg_maxiter=cg_maxiter, cg_tol=cg_tol, adam_maxiter=adam_maxiter, adam_tol=adam_tol)
	toc = time.process_time()
	
	"""
	Computing the MSE on the test data
	"""
	nlam = len(lam_list)
	mse_list = [] # to strore prediction error for each lam
	beta_list = []
	
	for i in range(nlam):
		model_final = model_list[i]

		X_hat = X_train[:, model_final]
		X_hatT = X_hat.T

		X_hatTy = X_hatT@y_train
		XX_hat = X_hatT@X_hat

		beta_hat = pinv(XX_hat)@X_hatTy 
		X_hat = X_test[:, model_final]
		mse = np.square(y_test - X_hat@beta_hat).mean()
		mse_list.append(mse)
		beta_pred = np.zeros(p)
		beta_pred[model_final] = beta_hat
		beta_list.append(beta_pred)

	ind_opt = np.argmin(mse_list)
	lam_opt = lam_list[ind_opt]
	model_opt = model_list[ind_opt]
	mse_opt = mse_list[ind_opt] 
	beta_opt = beta_list[ind_opt]
	
	return model_opt, mse_opt, beta_opt, lam_opt, toc - tic
