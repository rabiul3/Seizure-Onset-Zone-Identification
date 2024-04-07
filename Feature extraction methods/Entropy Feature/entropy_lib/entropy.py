# -*- coding: utf-8 -*-
#
# entropy2017.py
#
# Created by Kosuke FUKUMORI on 2018/02/02
#

import numpy as np
import itertools
from entropy_lib.entropy_tools import samp_entropy_core, ap_entropy_core, bispectral_entropy_core

def embed_seq(X, D, Tau=1):
	N = X.shape[0]
	return np.array([X[i:i+D*Tau] for i in range(N-(D-1)*Tau)], dtype=np.float32)

def samp_entropy(X, M, R, STD):
	N = X.shape[0]
	Em, Emp = embed_seq(X, M), embed_seq(X, M+1)
	Cm_mtrx, Cmp_mtrx = samp_entropy_core(Em, Emp[:,-1], R*STD)
	Cm = Cm_mtrx.sum() + 1e-100
	Cmp = Cmp_mtrx.sum() + 1e-100
	Samp_En = np.log(Cm/Cmp)
	return Samp_En

def ap_entropy(X, M, R, STD):
	N = X.shape[0]
	Em, Emp = embed_seq(X, M), embed_seq(X, M+1)
	Cm_mtrx, Cmp_mtrx, Cmi_vctr = ap_entropy_core(Em, Emp[:,-1], R*STD, N-M)
	Cm = Cm_mtrx.sum(0) + Cm_mtrx.sum(1) + Cmi_vctr
	Cmp = Cmp_mtrx.sum(0) + Cmp_mtrx.sum(1)
	Cm[N-M] += Cmi_vctr.sum() + 1
	Cm = Cm / (N - M + 1)
	Cmp = Cmp / (N - M)
	Phi_m, Phi_mp = np.log(Cm).sum(), np.log(Cmp).sum()
	Ap_En = (Phi_m - Phi_mp) / (N - M)
	return Ap_En

def spectral_entropy(X):
	fft_sig=np.fft.fft(X)
	print(fft_sig)
	Power = np.abs(np.fft.fft(X))
	print(Power)
	Power_Ratio = Power[:-1] / Power.sum()
	ShEn = -(Power_Ratio * np.log(Power_Ratio)).sum()
	RenEn = -np.log((Power_Ratio ** 2).sum())
	return ShEn, RenEn

def tsallis_entropy(X, order):
	Power = np.abs(np.fft.fft(X))
	Power_Ratio = Power / Power.sum()
	tsEn = (1 - (Power_Ratio ** order).sum()) / (order - 1)
	return tsEn

def bispectral_entropy(X):
	data_len = X.shape[0]
	Bispectral_len = (data_len // 4) * (data_len // 4 + 1)
	C = np.asarray(np.fft.fft(X)[:data_len//2])
	multipl_ratio = 10**int(-np.log10(np.abs(C).mean()))
	C = np.asarray(C * multipl_ratio, dtype=np.complex64)
	Bispectral = bispectral_entropy_core(C, C.shape[0], data_len, Bispectral_len)
	S1_p = Bispectral / Bispectral.sum()
	squared_Bispectral = Bispectral ** 2
	S2_q = squared_Bispectral / (squared_Bispectral).sum()
	#print(S2_q)
	S1 = -(S1_p[:-1] * np.log(S1_p[:-1])).sum() / np.log(S1_p.shape[0])
	S2 = -(S2_q[:-1] * np.log(S2_q[:-1])).sum() / np.log(S2_q.shape[0])
	return S1, S2

def permutation_entropy(X):
	Em = embed_seq(X, D=3)
	sort = np.argsort(Em)
	sort = sort[:,2]*4 + sort[:,1]*2 + sort[:,0] #indexing to unique number in 4,5,6,8,9,10
	pattern = np.unique(sort, return_counts=True)[1]
	pattern = pattern / sort.shape[0]
	pattern = pattern + (pattern == 0)
	PE = -(pattern * np.log2(pattern)).sum()
	return PE
