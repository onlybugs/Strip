# 22-7-7
# kli
# Deal memory is not enough.

import random
import numpy as np
import pandas as pd
import cooler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import scipy.signal as signal

def ReadFile(fn,chr):
    '''
    Read cooler
    '''
    rdata = cooler.Cooler(fn)
    rmat = rdata.matrix(balance=True).fetch(chr)
    rmat[np.isnan(rmat)] = 0
    
    return rmat
    
def GetOE(rmat):
    '''
    Get OE
    '''
    mat = rmat
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
    pca1 = PCA(n_components=3)
    pca1.fit(cor_oe)

    return oe,cor_oe,mask,pca1.components_

def Findmax(ab):    
    # x = np.array([1,8,9,5,-2,0,7,5,9,-6,1,2,-9,5])
    x = ab
    argx = signal.argrelextrema(x,np.greater)[0]
    maxidx = []
    for i in argx:
        if(x[i] > 0):
            maxidx.append(i)
    
    return maxidx

def DownSampling(rmat,ratio = 2):
    sampling_ratio = ratio
    # testm = rmat # np.arange(9).reshape(3,3)
    m = np.matrix(rmat)

    # 采样概率
    # all ele sum = 60
    all_sum = m.sum(dtype='float')
    # print("before sample :",all_sum)
    # !!!!必须要加
    m = m.astype(np.float64)
    # 其实就是除法
    idx_prob = np.divide(m, all_sum,out=np.zeros_like(m), where=all_sum != 0)
    # reshape 1,9
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    # 这是为了设计抽样概率
    idx_prob = np.squeeze(idx_prob)

    # 采样索引
    # 60 / 4
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    # 0 1 2 ... 8
    id_range = np.arange(m.shape[0]*m.shape[1])
    # np.random.seed(0)

    # choice 15 from id_range
    # p: idx_prob
    # 放回抽样 
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)

    # 输出矩阵
    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    # print("after sample :",sample_m.sum())

    return np.array(sample_m)

# 60%
def IdxFilter(ridx,ab):
    idxl = len(ridx)
    idxl = round(idxl * 0.6)

    AllPeak = ab[np.array(ridx)]
    Top60Peak = np.sort(AllPeak)[-idxl:]

    FilterOut = []
    for i in ridx:
        if(ab[i] in Top60Peak):
            FilterOut.append(i)
    
    return FilterOut
# Random
def FillupLen(mat,l):
    dif = l - mat.shape[0]
    if(dif % 2 == 0):
        up = dif // 2 
        down = up
    else:
        up = dif // 2 + 1
        down = up
    samples = np.random.choice(range(mat.shape[0]),dif) # random.sample(range(mat.shape[0]),dif)
    upfill = mat[np.array(samples[up:]),:]
    downfill = mat[np.array(samples[-down:]),:]
    # [up+mat_shape[0]+down,2] -> 5000,2
    out = np.vstack((upfill,mat,downfill))
    
    return out

def StackThreeLayer(rmat,oe,coe,sidx,InputLen):
    # rmat_ground_t = rmat[:,sidx:sidx+2]
    # rmat_ground_t = FillupLen(rmat_ground_t,InputLen)

    oe_ground_t = oe[:,sidx:sidx+2]
    oe_ground_t= FillupLen(oe_ground_t,InputLen)

    coe_ground_t = coe[:,sidx:sidx+2]
    coe_ground_t = FillupLen(coe_ground_t,InputLen)
    
    # (3,4896,2)
    # tmp = np.stack((rmat_ground_t,oe_ground_t,coe_ground_t),axis=0)
    tmp = np.stack((oe_ground_t,coe_ground_t),axis=0)

    return tmp

def Minmax(mat):
    mi = mat.min()
    ma = mat.max()
    return (mat - mi) / (ma-mi)
def Zscore(mat):
    a = mat
    return (a - np.mean(a))/np.std(a)
def Tov(mat,p):
    v = np.percentile(mat,p)
    mat[mat > v] = v

    return mat

chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22',"X" ]
# The longest chr is 4896 -> 5000 !
InputLen = 5000
fn = "/store/kli/workdir/compareABandTAD/data/Rao2014-GM12878-MboI-allreps-filtered.50kb.cool"
edgefn = "data/gm-50k-allchr-edge.npz"

# All chr
gm50k_all_S = []
gm50k_all_AE = []
gm50k_all_l = [] 
sample_stat = {"chr":[],"sam_n":[]}
# minmax_stat = {"type":[],"min":[],'max':[]}
for chr in chrs_list:
    # Step 1: Read rmat file and basic compute. Then we Sparse it and repeat forward.
    HicRmat = ReadFile(fn,"chr"+chr)
    oe,coe,mask,pc1 = GetOE(HicRmat)
    SparseRmat = DownSampling(HicRmat,ratio = 8)
    print(HicRmat.sum(),SparseRmat.sum())
    soe,scoe,smask,spc1 = GetOE(SparseRmat)
    oe = Tov(oe,95)
    soe = Tov(soe,95)
    print("oe max,soe max:",np.max(oe),np.max(soe))

    # Step 2: Read Edge file , get idx , filter it.
    EdgeValue = np.load(edgefn)[chr]
    StartEdgeIdx = Findmax(EdgeValue)
    Top60idx = IdxFilter(StartEdgeIdx,EdgeValue)

    # Step 3: positive samples. Sparse and dense sample is generated together.
    chrkTsample = []
    chrkTSsample = []
    label = []
    for sidx in Top60idx:
        chrkTsample.append(StackThreeLayer(HicRmat,oe,coe,sidx,InputLen))
        chrkTSsample.append(StackThreeLayer(SparseRmat,soe,scoe,sidx,InputLen))
        label.extend([1,1])

    # Step 4: negative samples. We use edge which not in ALL edge idx to a negative sample.
    chrlen = np.arange(EdgeValue.shape[0])
    chrlen = chrlen[mask]
    neg_idx = []
    # ...
    StartEdgeIdx.append(EdgeValue.shape[0]-1)
    for aa in chrlen:
        if(aa not in StartEdgeIdx):
            neg_idx.append(aa)
    chrkFsample = []
    chrkFSsample = []
    BanlanceNegIdx = np.random.choice(neg_idx,len(chrkTsample),replace=False)
    for nidx in BanlanceNegIdx:
        chrkFsample.append(StackThreeLayer(HicRmat,oe,coe,nidx,InputLen))
        chrkFSsample.append(StackThreeLayer(SparseRmat,soe,scoe,nidx,InputLen))
        label.extend([0,0])
    
    # Step 5:Combine. 
    sample_stat['chr'].append("chr" + chr)
    sample_stat['sam_n'].append(len(chrkTsample))
    print("chr :",chr,"  Sample num: ",len(chrkTsample))
    gm50k_all_S = gm50k_all_S + chrkTsample + chrkTSsample + chrkFsample + chrkFSsample
    # gm50k_all_AE = gm50k_all_AE + chrkTsample + chrkTSsample + chrkFsample + chrkFsample
    gm50k_all_l.extend(label)

# 33048  正负样本均衡
# Step 6:Shuffle data and train test split.
shuffle_idx = np.arange(len(gm50k_all_l),dtype=int)
# np.random.seed(0)
np.random.shuffle(shuffle_idx)
# gm50k_all_AE = np.stack(gm50k_all_AE,axis=0)[shuffle_idx,...]
gm50k_all_l = np.stack(gm50k_all_l,axis=0)[shuffle_idx]
gm50k_all_S = np.stack(gm50k_all_S,axis=0)[shuffle_idx,...]

# 7 2 1
train_n = int(gm50k_all_l.shape[0] * 0.7)
val_n = int(gm50k_all_l.shape[0] * 0.9)

# Step 7:Save
np.savez("train.npz",# AE = gm50k_all_AE[:train_n,...],
                     Sample = gm50k_all_S[:train_n,...],
                     label = gm50k_all_l[:train_n])

np.savez("valid.npz",# AE = gm50k_all_AE[train_n:val_n,...],
                     Sample = gm50k_all_S[train_n:val_n,...],
                     label = gm50k_all_l[train_n:val_n])
np.savez("test.npz",# AE = gm50k_all_AE[val_n:,...],
                     Sample = gm50k_all_S[val_n:,...],
                     label = gm50k_all_l[val_n:])
np.savez("cv.npz",# AE = gm50k_all_AE[val_n:,...],
                     Sample = gm50k_all_S[:val_n,...],
                     label = gm50k_all_l[:val_n])

sstat = pd.DataFrame(sample_stat)
sstat.to_csv("stat.csv")
