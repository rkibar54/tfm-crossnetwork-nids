import pandas as pd, numpy as np
from scipy.stats import wilcoxon
e8=pd.read_csv('results/e8_runs.csv'); e9=pd.read_csv('results/e9_runs.csv')
rng=np.random.default_rng(0)
def holm(ps):
    ps=np.asarray(ps); o=np.argsort(ps); n=len(ps); adj=np.empty(n); m=0
    for r,i in enumerate(o): m=max(m,(n-r)*ps[i]); adj[i]=min(1,m)
    return adj
def boot_pairs(v,n=10000):
    b=rng.choice(v,(n,len(v))).mean(1); return np.percentile(b,[2.5,97.5])
# ---------- RQ3 at pair level
pc=e8[(e8.method=='pooled-context')&(e8.model=='TabPFN')&(e8.k.isin([1,10]))]
w=pc.pivot_table(index=['source','target','seed','k'],columns='S',values=['bacc','f1'])
res=[]
for met in ['bacc','f1']:
    d=(w[(met,2000)]-w[(met,100)]).rename('d').reset_index()
    for kk in ['all',1,10]:
        x=d if kk=='all' else d[d.k==kk]
        pm=x.groupby(['source','target']).d.mean().values
        p=wilcoxon(pm).pvalue; ci=boot_pairs(pm)
        res.append((met,kk,pm.mean(),ci[0],ci[1],(pm<0).sum(),p))
R=pd.DataFrame(res,columns=['met','k','mean','lo','hi','n_neg','p']); R['holm']=holm(R.p)
print('RQ3 pair-level (12 pairs), S2000-S100'); print(R.round(4))
# also S500 vs S100, S2000 vs S500 pair-level BA
for a,b in [(500,100),(2000,500)]:
    d=(w[('bacc',a)]-w[('bacc',b)]).rename('d').reset_index(); pm=d.groupby(['source','target']).d.mean().values
    print(a,'-',b,'BA %.4f'%pm.mean(),boot_pairs(pm).round(4),'p=%.4f'%wilcoxon(pm).pvalue,'neg',(pm<0).sum())
# ---------- RQ2 recheck (pair level) already: report
# ---------- RQ1 target level, hierarchical bootstrap (targets, then seeds within)
A=e9[(e9.method=='target-only')&(e9.features=='all')]
CL=['RF','GBT','MLP','LightGBM','XGBoost','CatBoost']
def hboot(df,col,n=4000):
    # df indexed by target, seed; values col
    tg=df.target.unique(); g={t:df[df.target==t][col].values for t in tg}
    out=np.empty(n)
    for i in range(n):
        ts=rng.choice(tg,len(tg)); out[i]=np.mean([rng.choice(g[t],len(g[t])).mean() for t in ts])
    return np.percentile(out,[2.5,97.5])
for metric in ['bacc','pr_auc','roc_auc','f1']:
    P=A.pivot_table(index=['target','k','seed'],columns='model',values=metric)
    best=P.groupby(level=['target','k']).mean()[CL].idxmax(axis=1)
    orc=pd.Series([P.loc[i,best[(i[0],i[1])]] for i in P.index],index=P.index)
    for t in ['TabPFN','TabICL']:
        comps={'avg':P[CL].mean(1),'CatBoost':P['CatBoost'],'oracle':orc}
        if metric!='bacc': comps={'CatBoost':P['CatBoost']}
        if metric=='f1': comps={'avg3':P[['RF','GBT','MLP']].mean(1)}
        for cn,ref in comps.items():
            d=(P[t]-ref).rename('d').reset_index()
            for kk in ['all',1,5,10,50,100]:
                x=d if kk=='all' else d[d.k==kk]
                # average over k within target-seed for 'all'
                xs=x.groupby(['target','seed']).d.mean().reset_index()
                tm=xs.groupby('target').d.mean()
                ci=hboot(xs,'d')
                print(f'{metric:7s} {t} - {cn:8s} k={kk}: mean {tm.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] targets>0: {(tm>0).sum()}/4  per-target {tm.round(3).to_dict()}')
# TabPFN vs TabICL
P=A.pivot_table(index=['target','k','seed'],columns='model',values='bacc')
xs=(P.TabPFN-P.TabICL).rename('d').reset_index().groupby(['target','seed']).d.mean().reset_index()
print('TabPFN-TabICL',xs.groupby('target').d.mean().round(4).to_dict(),hboot(xs,'d'))
