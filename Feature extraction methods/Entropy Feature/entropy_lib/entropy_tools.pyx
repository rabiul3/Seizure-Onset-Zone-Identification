# -*- coding: utf-8 -*-
#
# entropy_pxx_lib.pyx
#
# Created by Kosuke FUKUMORI on 2018/02/03
#

import numpy as np
cimport numpy as np

ctypedef np.float32_t DTYPE_t

def check_last_one_matrix(np.ndarray[DTYPE_t, ndim=1] x, DTYPE_t distance):
  cdef int size_x = x.shape[0]
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] results = np.zeros([size_x, size_x], dtype=np.uint8)
  cdef int i, j
  for i in range(size_x):
    for j in range(i):
      results[i, j] = abs(x[i] - x[j]) <= distance
  return results

def in_range_matrix(np.ndarray[DTYPE_t, ndim=2] xs, DTYPE_t distance):
  cdef int size_xs = xs.shape[0]
  cdef int dim = xs.shape[1]
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] results = np.zeros([size_xs, size_xs], dtype=np.uint8)
  cdef int i, j, k
  for i in range(size_xs):
    for j in range(i):
      results[i, j] = 1
      for k in range(dim):
        results[i, j] = results[i, j] and (abs(xs[i,k] - xs[j,k]) <= distance)
  return results

def samp_entropy_core(np.ndarray[DTYPE_t, ndim=2] em, np.ndarray[DTYPE_t, ndim=1] emp_tail, DTYPE_t distance):
  cdef int dim = em.shape[1]
  cdef int size_emp = emp_tail.shape[0]
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] cm_mtrx = np.zeros([size_emp, size_emp], dtype=np.uint8)
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] cmp_mtrx = np.zeros([size_emp, size_emp], dtype=np.uint8)
  cdef int i, j, k
  for i in range(size_emp):
    for j in range(i):
      cm_mtrx[i, j] = 1
      for k in range(dim):
        cm_mtrx[i, j] = cm_mtrx[i, j] and (abs(em[i,k] - em[j,k]) <= distance)
      cmp_mtrx[i, j] = cm_mtrx[i, j] and (abs(emp_tail[i] - emp_tail[j]) <= distance)
  return cm_mtrx, cmp_mtrx

def ap_entropy_core(np.ndarray[DTYPE_t, ndim=2] em, np.ndarray[DTYPE_t, ndim=1] emp_tail, DTYPE_t distance, int nm):
  cdef int dim = em.shape[1]
  cdef int size_em = em.shape[0]
  cdef int size_emp = emp_tail.shape[0]
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] cm_mtrx = np.zeros([size_em, size_em], dtype=np.uint8)
  cdef np.ndarray[np.uint8_t, cast=True, ndim=2] cmp_mtrx = np.zeros([size_emp, size_emp], dtype=np.uint8)
  cdef np.ndarray[np.uint8_t, cast=True, ndim=1] cmi_vctr = np.zeros([size_em], dtype=np.uint8)
  cdef int i, j, k
  for i in range(size_emp):
    cm_mtrx[i, i] = 1
    cmp_mtrx[i, i] = 1
    for j in range(i):
      cm_mtrx[i, j] = 1
      for k in range(dim):
        cm_mtrx[i, j] = cm_mtrx[i, j] and (abs(em[i,k] - em[j,k]) <= distance)
      cmp_mtrx[i, j] = cm_mtrx[i, j] and (abs(emp_tail[i] - emp_tail[j]) <= distance)
    cmi_vctr[i] = 1
    for k in range(dim):
      cmi_vctr[i] = cmi_vctr[i] and (abs(em[i, k] - em[nm, k]) <= distance)
  cm_mtrx[nm, nm] = 1
  cmi_vctr[nm] = 1
  return cm_mtrx, cmp_mtrx, cmi_vctr

def bispectral_entropy_core(np.ndarray[np.complex64_t, ndim=1] c, int c_len, int data_len, int bispectral_len):
  cdef np.ndarray[np.complex64_t, ndim=1] c_conj = c.conj()
  cdef np.ndarray[DTYPE_t, ndim=2] bispect_mtrx = np.empty([data_len // 2, data_len // 4], dtype=np.float32)
  cdef np.ndarray[DTYPE_t, ndim=1] bispectral = np.empty([bispectral_len], dtype=np.float32)
  cdef int index = 0
  cdef int k, l, ddtok
  for k in range(data_len // 4):
    for l in range(k + 1):
      bispect_mtrx[k, l] = abs(c[k] * c[l] * c_conj[k+l])
      bispect_mtrx[data_len // 2 - 1 - k, l] = abs(c[data_len // 2 - 1 - k] * c[l] * c_conj[data_len // 2 - 1 - k+l])

  for k in range(data_len // 4):
    for l in range(k + 1):
      bispectral[index] = bispect_mtrx[k, l]
      bispectral[bispectral_len-1-index] = bispect_mtrx[data_len // 2 - 1 - k, l]
      index += 1
  return bispectral
