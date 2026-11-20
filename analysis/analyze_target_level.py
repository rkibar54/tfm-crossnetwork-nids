import pandas as pd, numpy as np, warnings
from scipy import stats
warnings.filterwarnings('ignore')
e7=pd.read_csv('results/e7_runs.csv'); e8=pd.read_csv('results/e8_runs.csv'); e9=pd.read_csv('results/e9_runs.csv')
def tci(tm):
    tm=np.asarray(tm,float); m=tm.mean(); h=stats.t.ppf(.975,len(tm)-1)*tm.std(ddof=1)/np.sqrt(len(tm)); return m,m-h,m+h
def fmt(tm): m,l,h=tci(tm); return f"{m:+.4f} [{l:+.4f},{h:+.4f}] pos {int((np.asarray(tm)>0).sum())}/4"
A=e9[(e9.method=='target-only')&(e9.features=='all')]
CL=['RF','GBT','MLP','LightGBM','XGBoost','CatBoost']; CL5=[c for c in CL if c!='XGBoost']
for met in ['bacc','f1','pr_auc','roc_auc']:
    P=A.pivot_table(index=['target','k','seed'],columns='model',values=met)
    best=A.pivot_table(index=['target','k','seed'],columns='model',values='bacc').groupby(level=['target','k']).mean()[CL].idxmax(axis=1)
    orc=pd.Series([P.loc[i,best[(i[0],i[1])]] for i in P.index],index=P.index)
    refs={'avg6':P[CL].mean(axis=1),'avg5':P[CL5].mean(axis=1),'avg3':P[['RF','GBT','MLP']].mean(axis=1),'Cat':P.CatBoost,'oracle':orc}
    for t in ['TabPFN','TabICL']:
        for rn,ref in refs.items():
            if met!='bacc' and rn in ('oracle','avg5'): continue
            d=P[t]-ref
            row=[f"all {fmt(d.groupby(level='target').mean())}"]
            for k in [1,5,10,50,100]:
                row.append(f"k{k} {fmt(d.xs(k,level='k').groupby(level='target').mean())}")
            print(met,t,rn,' | '.join(row))
    if met=='bacc':
        d=P.TabPFN-P.TabICL; print('TabPFN-TabICL',fmt(d.groupby(level='target').mean()))
# per-target PR-AUC diffs vs CatBoost at k50/100
P=A.pivot_table(index=['target','k','seed'],columns='model',values='pr_auc')
for t in ['TabPFN','TabICL']:
    for k in [50,100]:
        print('PR per target',t,k,(P[t]-P.CatBoost).xs(k,level='k').groupby(level='target').mean().round(4).to_dict())
# RQ2 target-level
def src(df,model,S=500,met='bacc',ks=None):
    d=df[df.model==model]
    to=d[d.method=='target-only'].drop_duplicates(['target','seed','k'])[['target','seed','k',met]]
    pc=d[(d.method=='pooled-context')&(d.S==S)]
    if ks: pc=pc[pc.k.isin(ks)]
    m=pc.merge(to,on=['target','seed','k'],suffixes=('','_to')); m['d']=m[met]-m[met+'_to']
    pairs=m.groupby(['source','target']).d.mean(); tg=pairs.groupby(level='target').mean()
    return pairs,tg
e7[['source','target']]=e7.direction.str.split('->',expand=True); e7['S']=500
def src7(met):
    d=e7[e7.model=='TabPFN']
    to=d[d.method=='target-only'][['source','target','seed','k',met]]
    pc=d[d.method=='pooled-context'].merge(to,on=['source','target','seed','k'],suffixes=('','_to'))
    pc['d']=pc[met]-pc[met+'_to']; pairs=pc.groupby(['source','target']).d.mean(); return pairs,pairs.groupby(level='target').mean()
for met in ['bacc','f1']:
    pairs,tg=src7(met); print('RQ2 E7 TabPFN',met,'pairs %.4f'%pairs.mean(),'target-level',fmt(tg),tg.round(3).to_dict())
for nm,df,mod in [('E8',e8,'TabPFN'),('E8',e8,'TabICL')]:
    for met in ['bacc','f1']:
        pairs,tg=src(df,mod,met=met)
        print('RQ2',nm,mod,met,'pairs %.4f'%pairs.mean(),'target-level',fmt(tg),tg.round(3).to_dict())
# RQ3 target-level
pc=e8[(e8.method=='pooled-context')&(e8.model=='TabPFN')&(e8.k.isin([1,10]))]
w=pc.pivot_table(index=['source','target','seed','k'],columns='S',values=['bacc','f1'])
for met in ['bacc','f1']:
    d=(w[(met,2000)]-w[(met,100)]).rename('d').reset_index()
    for kk in ['all',1,10]:
        x=d if kk=='all' else d[d.k==kk]
        tg=x.groupby(['source','target']).d.mean().groupby(level='target').mean()
        print('RQ3',met,kk,fmt(tg))
# TTL source effect target-level
n9=e9[(e9.method=='target-only')&(e9.features=='no-ttl')&(e9.model=='TabPFN')][['target','seed','k','bacc','f1']]
p9=e9[e9.method=='pooled-context'].merge(n9,on=['target','seed','k'],suffixes=('','_to'))
for met in ['bacc','f1']:
    p9['d']=p9[met]-p9[met+'_to']; tg=p9.groupby(['source','target']).d.mean().groupby(level='target').mean()
    print('TTL src',met,fmt(tg))
pairs,tg=src(e8,'TabPFN',ks=[1,10,100]); print('TTL all-feat same k',fmt(tg))
# k=50 table rows
k50=A[A.k==50].groupby('model')[['roc_auc','pr_auc','recall','precision','fpr']].mean()
print(k50.loc[CL+['TabPFN','TabICL']].round(3))
