# 22-7-13
# kli
# Main

import numpy as np
import pandas as pd

from Utils import GetHic,ABedge,MeanLoop,Findmax,IdxFilter
from MathCompute import svd

chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
             ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22',"X" ]
fn = "../data/Rao2014-GM12878-MboI-allreps-filtered.50kb.cool"
svdr = 1
decay = 0
# "data/Rao2014-K562-MboI-allreps-filtered.50kb.cool"
# "data/Rao2014-GM12878-MboI-allreps-filtered.mcool::resolutions/20000"

outnpz_edge = {}
outnpz_pc = {}
edgen = {}
edgen['fun'] = ['ours','pca']
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

    # 记录长度
    edgen[chnum].extend([len(filteridx),len(pc_idx)])

    print("chr " + chnum + " is :",edgen[chnum])

edgen = pd.DataFrame(edgen)
edgen.to_csv("gm50k-filter.csv")

# np.savez("gm-edgeidx-75.npz",**outnpz_edge)
# np.savez("gm-pcidx.npz",**outnpz_pc)
# np.savez("gm-raoidx.npz",**outnpz_rao)

print("All is done! ^_^")
