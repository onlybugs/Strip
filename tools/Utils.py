# 22-7-13
# kli
# Extra Fun

from HicObj import HicObj
import numpy as np
from MathCompute import Findmax,adaptive_window_mean

def GetHic(fn,chr):
    rhic = HicObj()
    rhic.ReadFile(fn,chr)
    rhic.GetOE()
    rhic.DI_INS()
    rhic.ABedge()
    return rhic

def DelLast(edge,dratio = 0.05):
    idx = Findmax(edge)
    dnum = int(len(idx) * dratio)
    peakVal = edge[idx]

    peakVal = np.sort(peakVal)
    dpeak = peakVal[:dnum]
    
    outidx = []
    for i in idx:
        if(edge[i] not in dpeak):
            outidx.append(i)

    return outidx

def ABedge(mat):
    testd = mat
    two = np.zeros_like(testd)
    three = np.zeros_like(testd)
    four = np.zeros_like(testd)

    _len = testd.shape[0]
    for i in range(1,_len):
        three[i,:] = testd[i-1,:]
        four[:,i-1] = testd[:,i]
        two[i,0:_len-1] = testd[i-1,1:_len]
    edgem = three * four - testd * two
    out = np.sum(np.abs(edgem),0)
    # 减去均值
    out -= out.mean()
    # 取所有负值的分位点
    _mab = out[out < 0]
    p = np.percentile(_mab,85)
    # 调整
    out = out + np.abs(p)
    
    return out

def MeanLoop(NewOE,d,redge,mask):

    loopoe = NewOE
    dedge = redge
    decay = d
    while(decay > 0):
        dedge = DelLast(dedge,dratio = decay)
        adp_mat = adaptive_window_mean(loopoe.shape[0],dedge)
        mat = np.matrix(adp_mat) * np.matrix(loopoe) + (np.matrix(adp_mat) * np.matrix(loopoe)).T
        cor_oe = np.corrcoef(mat)
        cor_oe[np.isnan(cor_oe)] = 0
        cor_oe[~mask,:] = 0
        cor_oe[:,~mask] = 0
        dedge = ABedge(cor_oe)

        loopoe = mat
        decay -= 0.01

    return dedge

def IdxFilter(ridx,ab,k = 0.75):
    idxl = len(ridx)
    idxl = round(idxl * k)

    AllPeak = ab[np.array(ridx)]
    Top60Peak = np.sort(AllPeak)[-idxl:]

    FilterOut = []
    for i in ridx:
        if(ab[i] in Top60Peak):
            FilterOut.append(i)
    
    return FilterOut
