# 22-7-13
# HicObj

import numpy as np
import pandas as pd
from scipy.signal import convolve
import cooler
from sklearn.decomposition import PCA
from sklearn import cluster


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
