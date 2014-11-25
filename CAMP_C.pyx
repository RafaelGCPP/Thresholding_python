# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:14:52 2014

@author: Rafael
"""
import numpy as np
import scipy.linalg.lapack as lpk
from cvxopt import matrix
cimport numpy as np
cimport cython

np.import_array()

ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
ctypedef complex64_t COMPLEX64
cdef int FORTRAN = 1

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL


from blas_types cimport dgemm_t, zgemm_t, ddot_t, dgemv_t, zgemv_t, zdotu_t

def get_func(name,dtype):
	return scipy.linalg.blas.get_blas_funcs(name, dtype)._cpointer
	
cdef dgemm_t *dgemm = <dgemm_t*>f2py_pointer(get_func('gemm', float64))
cdef zgemm_t *zgemm = <zgemm_t*>f2py_pointer(get_func('gemm', complex128))
cdef  ddot_t *ddot  = <ddot_t*> f2py_pointer(get_func('dot', float64))
cdef dgemv_t *dgemv = <dgemv_t*>f2py_pointer(get_func('gemv', float64))
cdef zdotu_t *zdotu = <zdotu_t*>f2py_pointer(get_func('dotu', complex128))
cdef zgemv_t *zgemv = <zgemv_t*>f2py_pointer(get_func('gemv', complex128))

def STc(x,theta,copy=True):
    if copy:    
        z=x.copy()
    else:
        z=x
    az=abs(z)
    z[az<theta]=0
    z=z-theta*(z/az)    
    return z
    
    
def dSTc(x,theta):
    eps=1e-12
    z=x.copy()
    az=abs(z)
    az3=az**3+eps
    x=real(z)
    y=imag(z)
    
    d1R=1-(theta*y**2)/az3
    d2I=1-(theta*x**2)/az3

    d1R[az<theta]=0
    d2I[az<theta]=0
      
    return (d1R,d2I)   
	
def CAMP(A,y,beta,verbose=False):
    M,N=A.shape
    x_old=zeros((N,1))
    z=y.copy()  
    eps=1e-12
    it=0
    while True:
        tz=dot(A.T.conj(),z)+x_old
        sigma_hat=beta*1/sqrt(2)*median(abs(tz))
        x=STc(tz,sigma_hat)
        (dR,dI)=dSTc(tz,sigma_hat)
        z=y-dot(A,x)+z*(sum(dR)+sum(dI))/(2*N)
        n=norm(x-x_old,2)
        if n<eps*norm(x,2):
            break
        if verbose:
            it+=1
            print "AMP iteration: %d (error %g)"%(it,n)
        x_old=x

    return x