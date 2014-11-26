# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:45:09 2014

@author: Rafael
"""
from numpy import argsort,sign,abs,zeros
from numpy import sum,sqrt,dot,real,imag,median
from numpy.linalg import norm

#Thresholding operators; in matrices operates over columns

def HT_t(x,theta,copy=True):
    if copy:    
        ret=x.copy()
    else:
        ret=x
    ret[abs(ret)<theta]=0
    return ret

def HT_k(x,k,copy=True):
    if copy:    
        ret=x.copy()
    else:
        ret=x
    s=argsort(abs(ret),axis=0)
    s=s[::-1,:]
    for (i,c) in enumerate(ret.T): # iterate over columns
        c[s[k:,i]]=0
    return ret

def ST(x,theta,copy=True):
    if copy:    
        ret=x.copy()
    else:
        ret=x
    s=sign(ret)
    ret=abs(ret)-theta
    ret[ret<0]=0
    ret=s*ret
    return ret

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
#    d2R=x*y*theta/az3
#    d1I=x*y*theta/az3
    d2I=1-(theta*x**2)/az3

    d1R[az<theta]=0
#    d2R[az<theta]=0
#    d1I[az<theta]=0
    d2I[az<theta]=0
    
#    return (d1R,d2R,d1I,d2I)    
    return (d1R,d2I)    
    
def AMP2(A,y,beta,verbose=False):
    M,N=A.shape
    x_old=zeros((N,1))
    z=y.copy()  
    eps=1e-12
    it=0
    while True:
        c=1.0/M*sum(z*z)
        theta=sqrt(beta*c)
        x=ST(x_old+dot(A.T,z),theta)
        card=sum(abs(x)>theta)
        z=y-dot(A,x)-1/M*card
        n=norm(x-x_old,2)
        if n<eps*norm(x,2):
            break
        if verbose:
            it+=1
            print "AMP iteration: %d (error %g, %g)"%(it,n,theta)
        x_old=x
    if verbose:
        it+=1
        print "AMP iteration: %d (error %g, %g)"%(it,n,theta)
        
    return x

def CAMP(A,y,beta,verbose=False):
    M,N=A.shape
    x_old=zeros((N,1))
    z=y.copy()  
    eps=1e-12
    it=0
    while True:
        tz=dot(A.T.conj(),z)+x_old
        sigma_hat=beta*1/sqrt(2)*median(abs(tz))
#        sigma_hat=sqrt(beta*1.0/M*sum(z*z))
        x=STc(tz,sigma_hat)
        (dR,dI)=dSTc(tz,sigma_hat)
        z=y-dot(A,x)+z*(sum(dR)+sum(dI))/(2*N)
        n=norm(x-x_old,2)
        if it>1000:
            break
        if n<=eps*norm(x,2):
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

def main():
    from matplotlib.pyplot import figure,plot, close
    from numpy.random import standard_normal,choice
    from numpy.linalg import qr
    from numpy import dot
    import CAMP_C
    #from myOmp import omp_naive as omp
    N=2000
    M=900
    K=100
    sigma_n=0.001
    A=standard_normal((N,N))+1j*standard_normal((N,N))
    (Q,R)=qr(A)
    i=choice(N,M,False)  
    A=Q[i,:]

    x=(standard_normal((N,1))+1j*standard_normal((N,1)))/sqrt(2)
    j=choice(N,N-K,False)
    x[j,:]=0
    
    y=dot(A,x)+sigma_n*standard_normal((M,1))
    xhat=CAMP_C.CAMP(A,y,1,True)
    print norm(x-xhat)/N
    close('all')
    plot(real(x))
    plot(real(xhat))
    figure()
    plot(imag(x))
    plot(imag(xhat))
        
if __name__=="__main__":
    main() 