# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:14:52 2014

@author: Rafael
"""
import numpy as np
import numpy.linalg as la
import scipy.linalg.blas 

from cvxopt import matrix
cimport numpy as np
cimport cython

from blas_types cimport dgemm_t, zgemm_t, ddot_t, dgemv_t, zdotu_t, zgemv_t
from numpy cimport PyArray_ZEROS

from numpy cimport float64_t, ndarray, complex128_t, complex64_t
from numpy import float64, ndarray, complex128, complex64

ctypedef float64_t DOUBLE
ctypedef complex128_t dcomplex
ctypedef complex64_t COMPLEX64
cdef int FORTRAN = 1

cdef extern from "f2pyptr.h":
    void *f2py_pointer(object) except NULL



np.import_array()


def get_func(name,dt):
    return scipy.linalg.blas.get_blas_funcs(name, dtype=dt)._cpointer


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
    az=np.abs(z)
    az3=az**3+eps
    x=np.real(z)
    y=np.imag(z)
    
    d1R=1-(theta*y**2)/az3
    d2I=1-(theta*x**2)/az3

    d1R[az<theta]=0
    d2I[az<theta]=0
      
    return (d1R,d2I)   
	
 
def CAMP(A,y,beta,verbose=False):
    A=np.asfortranarray(A)
    y=np.asfortranarray(y)
    M,N=A.shape
    return CAMP_impl(A,y,beta,M,N,verbose)
    
cdef CAMP_impl(dcomplex[::1,:] A, 
               dcomplex[::1,:] y, 
               double beta,
               int M,
               int N,
               int verbose):
    x_old=np.zeros((N,1))
    z=y.copy()  
    eps=1e-12
    it=0
    beta2=beta*1/np.sqrt(2)
    while True:
        tz=np.dot(A.T.conj(),z)+x_old
        sigma_hat=beta2*np.median(abs(tz))
        x=STc(tz,sigma_hat)
        (dR,dI)=dSTc(tz,sigma_hat)
        z=y-np.dot(A,x)+z*(np.sum(dR)+np.sum(dI))/(2*N)
        n=la.norm(x-x_old,2)
        if it>1000:
            break
        if n<=eps*la.norm(x,2):
            break
        if verbose:
            it+=1
            if (it%10)==0:
                print "AMP iteration: %d (error %g, %g)"%(it,n,sigma_hat)
        x_old=x
    if verbose:
        it+=1
        print "AMP iteration: %d (error %g, %g)"%(it,n,sigma_hat)
     
    return x
