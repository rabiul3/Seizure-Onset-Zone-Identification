# -*- coding: utf-8 -*-

import numpy as np
try:
	import cupy as cp
except:
	import numpy as cp

def embed_seq(X, D, Tau=1):
	N = X.shape[0]
	return np.array([X[i:i+D*Tau] for i in range(N-(D-1)*Tau)], dtype=np.float32)

def in_range(Template, Scroll, Distance):
	return (np.abs(Template-Scroll[:Template.shape[0]]) > Distance).sum() == 0

def in_range_matrix(Xs, Distance):
	dim = Xs.shape[1]
	abs_dist = np.abs(Xs.reshape([-1,1,dim]) - Xs.reshape([1,-1,dim]))
	return (abs_dist > Distance).sum(axis=2) == 0

def check_last_one_matrix(X, Distance):
	return np.abs(X.reshape([-1,1]) - X) <= Distance

def samp_entropy(X, M, R):
	# Computer sample entropy (SampEn) of series X, specified by M and R.
	N = X.shape[0]

	Em, Emp = embed_seq(X, M), embed_seq(X, M+1)


	in_ran_mat = np.tril(in_range_matrix(Em, R), k=-1)
	Cm = in_ran_mat.sum(0) + 1e-100
	Cmp = np.logical_and(in_ran_mat[:-1,:-1], check_last_one_matrix(Emp[:,-1], R)).sum(0) + 1e-100
	Samp_En = np.log(Cm.sum()/Cmp.sum())

	return Samp_En
