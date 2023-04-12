# 22-7-13
# kli
# ...

import numpy as np
import scipy.signal as signal

def svd(HicObj,porn):
    '''
    HicObj:a hic obj
    porn: p or num,p is (0-1) and num > 1.if porn = 1,svd do nothing!
    '''
    U,s,V = np.linalg.svd(HicObj.oe)
    i = int(s.shape[0] * porn)
    A = np.dot(U[:,0:i],np.dot(np.diag(s[0:i]),V[0:i,:]))

    return A

def adaptive_window_mean(length, max_site):  
    mid_sites = np.arange(length)
    mat = np.zeros((length, length))
  
    start = 0 
    end = max_site[0]
    j = 0
    for i,x in enumerate(mid_sites):

        lens = end - start + 1
        mat[i, start : end + 1] = 1.0 / lens

        if end == x :
            start = end 
            j+= 1
            end = max_site[j]
        
        elif end == max_site[-1] :
            start = end
            end = length
    return mat

def Findmax(ab):
    # x = np.array([1,8,9,5,-2,0,7,5,9,-6,1,2,-9,5])
    x = ab
    argx = signal.argrelmax(x)[0]
    maxidx = []
    for i in argx:
        if(x[i] > 0):
            maxidx.append(i)
    
    return maxidx


