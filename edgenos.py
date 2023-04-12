# 22-3-16
# kli
# ...

import numpy as np
import pandas as pd
from scipy.signal import convolve
import cooler
from sklearn.decomposition import PCA
from sklearn import cluster
import scipy.signal as signal

class HicObj(object):
    def __init__(self):
        self.name = None
        self.chr = None
        self.rmat = None
        self.mask = None
        self.oe = None
        self.coe = None
        self.ins = None
        self.pca = None
        self.abedge = None

    def ReadFile(self,fn,chr):
        '''
        Read cooler
        '''
        rdata = cooler.Cooler(fn)
        self.name = fn
        self.chr = chr
        rmat = rdata.matrix(balance=True).fetch(chr)
        rmat[np.isnan(rmat)] = 0
        self.rmat = rmat
    
    def GetOE(self):
        '''
        Get OE
        '''
        mat = self.rmat
        chr_len = mat.shape[0]
        cut_off = chr_len/10
        mask = np.zeros(chr_len)
        num_mat = mat.copy()
        num_mat[num_mat > 0] = 1
        num_vector = np.sum(num_mat,axis=0)
        for i in range(chr_len):
            if(num_vector[i] >= cut_off):
                mask[i] = 1
        mask = mask == 1

        ox = np.arange(chr_len)
        oy = np.arange(chr_len)
        omask = mask.copy()
        decay = {}
        for i in range(chr_len):
            o_diag = mat[(ox,oy)]
            o_diag_mask = o_diag[omask]
            # gap
            if(o_diag_mask.shape[0] == 0):
                decay[i] = 0
            else:
                decay[i] = o_diag_mask.mean()
            ox = np.delete(ox,-1)
            oy = np.delete(oy,0)
            omask = np.delete(omask,-1)

        ex = np.arange(chr_len)
        ey = np.arange(chr_len)
        except_mat = np.ones_like(mat,dtype = np.float32)
        for i in range(chr_len):
            if(decay[i] == 0):
                ex = np.delete(ex,-1)
                ey = np.delete(ey,0)
                continue
            except_mat[(ex,ey)] = decay[i]
            except_mat[(ey,ex)] = decay[i]
            ex = np.delete(ex,-1)
            ey = np.delete(ey,0)
            
        oe = mat/except_mat
        cor_oe = np.corrcoef(oe)
        cor_oe[np.isnan(cor_oe)] = 0

        self.oe = oe
        self.coe = cor_oe
        self.mask = mask
        pca1 = PCA(n_components=3)
        pca1.fit(cor_oe)
        self.pca = pca1.components_

    def DI_INS(self):
        n = 10
        mat = self.rmat
        insul_cn_a = np.zeros((n,n))
        insul_cn_total = np.ones((n,n))
        DI_cn_U = np.zeros((n,n))
        DI_cn_D = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i >= n/2 and j >= n/2:
                    DI_cn_U[i,j] = 1
                elif i < n/2 and j < n/2:
                    DI_cn_D[i,j] = 1
                elif i >= n/2 and j < n/2:
                    insul_cn_a[i,j] = 1
                
        for i in range(n):
            for j in range(n):            
                if i<j:
                    insul_cn_total[i,j]=0
                    DI_cn_U[i,j]=0
                    DI_cn_D[i,j]=0
                    
        total = convolve(mat, insul_cn_total, mode = 'same')
        a = convolve(mat, insul_cn_a, mode = 'same')
        ins = np.log2(np.abs(np.diag(a) / np.diag(total)))*-1
        p95 = np.percentile(ins,95)
        p5 = np.percentile(ins,5)
        ins[ins> p95] = p95
        ins[ins< p5] = p5

        U = np.diag(convolve(mat, DI_cn_U, mode = 'same'))
        D = np.diag(convolve(mat, DI_cn_D, mode = 'same'))
        DI = (U-D)**3/(U+D)
        p95 = np.percentile(DI,95)
        p5 = np.percentile(DI,5)
        DI[DI> p95] = p95
        DI[DI< p5] = p5
        
        ins = ins - np.mean(ins)
        self.ins = ins

    def ABedge(self):
        testd = self.coe
        two = np.zeros_like(testd)
        three = np.zeros_like(testd)
        four = np.zeros_like(testd)

        _len = testd.shape[0]
        for i in range(1,_len):
            three[i,:] = testd[i-1,:]
            four[:,i-1] = testd[:,i]
            two[i,0:_len-1] = testd[i-1,1:_len]
        edgem = testd * two - three * four
        out = np.sum(np.abs(edgem),0)
        # 减去均值
        out -= out.mean()
        # 取所有负值的分位点
        _mab = out[out < 0]
        p = np.percentile(_mab,85)
        # 调整
        out = out + np.abs(p)
        self.abedge = out

def GetHic(fn,chr):
    rhic = HicObj()
    rhic.ReadFile(fn,chr)
    rhic.GetOE()
    rhic.DI_INS()
    rhic.ABedge()
    return rhic

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

def Rao_Idx(hico):
    cls_ab = cluster.KMeans(n_clusters=6)
    cls_ab.fit(hico.oe)
    lab = cls_ab.labels_
    # lab.shape
    lab[~hico.mask] = -1
    Rao_idx = [] # np.zeros(lab.shape[0])
    for i in range(lab.shape[0]-1):
        if(lab[i] >= 0 and lab[i+1] >= 0 and lab[i] != lab[i+1]):
            Rao_idx.append(i)

    return np.array(Rao_idx)

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


chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22',"X" ]
fn = "data/Rao2014-GM12878-MboI-allreps-filtered.50kb.cool"
svdr = 1
decay = 0
# "data/Rao2014-K562-MboI-allreps-filtered.50kb.cool"
# "data/Rao2014-GM12878-MboI-allreps-filtered.mcool::resolutions/20000"

outnpz_edge = {}
outnpz_pc = {}
outnpz_rao = {}
edgen = {}
edgen['fun'] = ['ours','pca','cluster']
for chnum in chrs_list:
    print("Deal to chr :",chnum)
    edgen[chnum] = []

    hico = GetHic(fn,'chr'+chnum)
    # OE 以及 SVD 作为第一个步骤
    NewOE = svd(hico,svdr)
    cNewOE = np.corrcoef(NewOE)
    cNewOE[np.isnan(cNewOE)] = 0
    cNewOE[~hico.mask,:] = 0
    cNewOE[:,~hico.mask] = 0
    raw_edge = ABedge(cNewOE)
    # ours
    edge_out = MeanLoop(NewOE,decay,raw_edge,hico.mask)
    edgeidx = Findmax(edge_out)

    filteridx = IdxFilter(edgeidx,edge_out)

    outnpz_edge[chnum] = np.array(filteridx)

    # pca
    pc1 = hico.pca[0,:]
    pc_idx = []
    for i in range(pc1.shape[0]-1):
        if(pc1[i] * pc1[i+1] < 0):
            pc_idx.append(i)
    outnpz_pc[chnum] = np.array(pc_idx)

    # Cluster
    outnpz_rao[chnum] = Rao_Idx(hico)

    # 记录长度
    edgen[chnum].extend([len(filteridx),len(pc_idx),outnpz_rao[chnum].shape[0]])

    print("chr " + chnum + " is :",edgen[chnum])

edgen = pd.DataFrame(edgen)
edgen.to_csv("gm50k-filter.csv")

np.savez("gm-edgeidx-75.npz",**outnpz_edge)
# np.savez("gm-pcidx.npz",**outnpz_pc)
# np.savez("gm-raoidx.npz",**outnpz_rao)

print("All is done! ^_^")
