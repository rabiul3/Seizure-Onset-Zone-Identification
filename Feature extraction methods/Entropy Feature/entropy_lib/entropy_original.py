# -*- coding: utf-8 -*-

import numpy as np




from numpy import zeros, log, log2
def embed_seq(X,Tau,D):

	N =len(X)

	if D * Tau > N:
		print ("Cannot build such a matrix, because D * Tau > N")
		exit()

	if Tau<1:
		print ("Tau has to be at least 1")
		exit()

	Y=zeros((N - (D - 1) * Tau, D))
	for i in range(0, N - (D - 1) * Tau):
		for j in range(0, D):
			Y[i][j] = X[i + j * Tau]
	return Y

def in_range(Template, Scroll, Distance):

	for i in range(0,  len(Template)):
			if abs(Template[i] - Scroll[i]) > Distance:
			     return False
	return True

def samp_entropy(X, M, R):

	N = len(X)

	Em = embed_seq(X, 1, M)
	Emp = embed_seq(X, 1, M + 1)

	Cm, Cmp = zeros(N - M - 1) + 1e-100, zeros(N - M - 1) + 1e-100

	for i in range(0, N - M):
		for j in range(i + 1, N - M):
			if in_range(Em[i], Em[j], R):
				Cm[i] += 1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R:
					Cmp[i] += 1

	Samp_En = log(sum(Cm)/sum(Cmp))

	return Samp_En

def ap_entropy(X, M, R):

	N = len(X)

	Em = embed_seq(X, 1, M)
	Emp = embed_seq(X, 1, M + 1)

	Cm, Cmp = zeros(N - M + 1), zeros(N - M)


	for i in range(0, N - M):

		for j in range(i, N - M):

			if in_range(Em[i], Em[j], R):
				Cm[i] += 1
				Cm[j] += 1
				if abs(Emp[i][-1] - Emp[j][-1]) <= R:
					Cmp[i] += 1
					Cmp[j] += 1
		if in_range(Em[i], Em[N-M], R):
			Cm[i] += 1
			Cm[N-M] += 1


	Cm[N - M] += 1

	Cm /= (N - M +1 )
	Cmp /= ( N - M )

	Phi_m, Phi_mp = sum(log(Cm)),  sum(log(Cmp))

	Ap_En = (Phi_m - Phi_mp) / (N - M)

	return Ap_En



from numpy.fft import fft

def spectral_entropy(X):

	Power = abs(fft(X))
	Power_Ratio = Power/sum(Power)

	ShEn = 0
	RenEn = 0
	for i in range(0, len(Power_Ratio) - 1):
		ShEn -= Power_Ratio[i] * log(Power_Ratio[i])
		RenEn += Power_Ratio[i]**2
	RenEn = -log(RenEn)

	return ShEn, RenEn
from numpy.fft import fft

def tsallis_entropy(X,order):

	Power = abs(fft(X))
	Power_Ratio = Power/sum(Power)
	tsEn = 1/(order - 1) * (1 - (Power_Ratio**order).sum())
	return tsEn



def bispectral_entropy(X):
	data_len = len(X)
	Bispectral_len = data_len/4*(data_len/4+1)
	C = fft(X)
	C = C[:data_len/2]

	index = 0
	Bispectral = [0 for col in range(Bispectral_len)]
	for k in range(data_len/4):
		for l in range(k+1):
			Bispectral[index] = C[k]*C[l]*np.conjugate(C[k+l])
			Bispectral[Bispectral_len-1-index] = C[data_len/2-1-k]*C[l]*np.conjugate(C[data_len/2-1-k+l])
			index += 1

	Bispectral = abs(np.array(Bispectral))
	S1_p = Bispectral / sum(Bispectral)
	S2_q = Bispectral**2 / sum(Bispectral**2)

	S1 = 0
	S2 = 0
	for i in range(0, len(S1_p) - 1):
		S1 -= S1_p[i] * log(S1_p[i])
		S2 -= S2_q[i] * log(S2_q[i])
	S1 = S1 / log(len(S1_p))
	S2 = S2 / log(len(S2_q))
	return S1, S2


def permutation_entropy(X):


	Em = embed_seq(X, 1, 3)
	sort = np.argsort(Em)
	sort = sort.tolist()

	pattern=[None for col in range(6)]
	pattern[0]=sort.count([0,1,2])
	pattern[1]=sort.count([0,2,1])
	pattern[2]=sort.count([1,0,2])
	pattern[3]=sort.count([1,2,0])
	pattern[4]=sort.count([2,0,1])
	pattern[5]=sort.count([2,1,0])

	lx=len(sort)
	lx+=0.
	PE=0.
	for k in pattern:
		k /= lx
		if k==0:
			k=1
		PE -= k*log2(k)
	return PE

def fuzzy_entropy(X,M,R,n):
	N = len(X)
	D_b = np.zeros(M)
	phi = np.zeros(M)
	d_a = np.zeros(M)
	D_a = np.zeros(M)
	D_mb = np.zeros(M)
	phia = np.zeros(M)
	d_ma = np.zeros(M)
	D_ma = np.zeros(M)
	Em = embed_seq(X, 1, M)
	Emp = embed_seq(X, 1, M + 1)

	for i in range(N - M):
		phi =phi + 1 / (N - M) * D_b
		for j in range(N - M):
			D_b = D_b + 1 / (N - M - 1) * D_a
			for k in range(M):
				d_a = abs((Em[i + k] - Em[i:].mean() / M) - (Em[j + k] - Em[j:].mean() / M)).max()
				D_a = np.exp(- (d_a ** n ) / R)
				print(D_a)


	for i in range(N - M):
		phia = phia + 1 / (N - M) * D_mb

		for j in range(N - M):
			D_mb = D_mb + 1 / (N - M - 1) * D_ma
			for k in range(M):
				d_ma = abs((Emp[i + k] - Emp[i:].mean() / M) - (Emp[j + k] - Emp[j:].mean() / M)).max()
				D_ma = np.exp(- (d_ma ** n ) / R)

	fuzzy = log(phi) - log(phia)
