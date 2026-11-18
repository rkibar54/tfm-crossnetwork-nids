import pandas as pd, numpy as np
from scipy.stats import wilcoxon
e9=pd.read_csv('results/e9_runs.csv'); e8=pd.read_csv('results/e8_runs.csv')
CL=['RF','GBT','MLP','LightGBM','XGBoost','CatBoost']; TF=['TabPFN','TabICL']
to=e9[(e9.method=='target-only')]
A=to[to.features=='all']
pd.set_option('display.width',250)
print('== BA / F1 by k (mean over 4 targets x 10 seeds)')
for m in ['bacc','f1','roc_auc','pr_auc','recall','precision','fpr']:
    print(m); print(A.groupby(['k','model'])[m].mean().unstack()[CL+TF].round(3))
def bh(p,q=.05):
    p=np.asarray(p); n=len(p); o=np.argsort(p); r=np.empty(n,bool)
    th=q*np.arange(1,n+1)/n; ok=p[o]<=th; kmax=np.where(ok)[0].max()+1 if ok.any() else 0
    r[:]=False; r[o[:kmax]]=True; return r
print('== per-cell TFM vs each classical (target x k), BA, raw and BH, family = 6 models x 20 cells = 120')
for t in TF:
    rows=[]
    for c in CL:
        for (tg,k),g in A.groupby(['target','k']):
            x=g[g.model==t].sort_values('seed').bacc.values; y=g[g.model==c].sort_values('seed').bacc.values
            dd=x-y
            p=1.0 if np.allclose(dd,0) else wilcoxon(x,y).pvalue
            rows.append((c,tg,k,dd.mean(),p))
    R=pd.DataFrame(rows,columns=['c','tg','k','d','p']); R['bh']=bh(R.p)
    for c in CL:
        r=R[R.c==c]; print(t,'vs',c,'raw better/worse',((r.p<.05)&(r.d>0)).sum(),((r.p<.05)&(r.d<0)).sum(),
              '| BH',((r.bh)&(r.d>0)).sum(),((r.bh)&(r.d<0)).sum(),'| mean d %.4f'%r.d.mean())
print('== TFM minus avg-6 / oracle-6 / best booster, BA by k, CI over target x seed cells')
rng=np.random.default_rng(0)
def ci(v):
    b=[rng.choice(v,len(v)).mean() for _ in range(5000)]; return np.mean(v),*np.percentile(b,[2.5,97.5])
piv=A.pivot_table(index=['target','k','seed'],columns='model',values='bacc')
mean_tk=A.groupby(['target','k','model']).bacc.mean().unstack()
orc=mean_tk[CL].idxmax(axis=1); print('oracle picks:',orc.value_counts().to_dict())
for t in TF:
    for k in [1,5,10,50,100,'all']:
        P=piv if k=='all' else piv.xs(k,level='k',drop_level=False)
        avg=P[CL].mean(axis=1); o=np.array([P.loc[i,orc[(i[0],i[1])]] for i in P.index])
        a=ci((P[t]-avg).values); b=ci((P[t].values-o)); 
        ob=mean_tk[['LightGBM','XGBoost','CatBoost']].idxmax(axis=1)
        print(t,k,'vsAvg %.3f [%.3f,%.3f]'%a,'vsOracle %.3f [%.3f,%.3f]'%b, 'p=%.3g'%wilcoxon(P[t].values-o).pvalue)
print('== per target k=10 BA all models'); print(mean_tk.xs(10,level='k')[CL+TF].round(3))
print('== per target k=1,100'); print(mean_tk.xs(1,level='k')[CL+TF].round(3)); print(mean_tk.xs(100,level='k')[CL+TF].round(3))
# confusion
print('== confusion k=10 summed over seeds, per target (TabICL, TabPFN, RF, CatBoost)')
c=A[A.k==10].groupby(['target','model'])[['tp','fp','tn','fn']].sum()
c['fpr']=c.fp/(c.fp+c.tn); c['fnr']=c.fn/(c.fn+c.tp)
print(c.loc[(slice(None),['RF','CatBoost','TabPFN','TabICL']),:].round(3))
# timings
print('== timing, median per call (20k test), all features')
T=A.copy(); T['thr']=T.n_test/T.pred_sec
print(T.groupby(['model','k'])[['fit_sec','pred_sec','thr']].median().unstack('k').round(3).loc[CL+TF][[('fit_sec',10),('fit_sec',100),('pred_sec',10),('pred_sec',100),('thr',10),('thr',100)]])
print(T.groupby('model')[['fit_sec','pred_sec']].agg(['median','max']).round(2).loc[CL+TF])
print(T.groupby('model').device.first())
L=e9.dropna(subset=['lat1_ms']); print(L.groupby('model')[['lat1_ms','lat1000_ms']].median().round(1).loc[CL+TF])
print(L.groupby(['model','target'])[['lat1_ms','lat1000_ms']].median().round(1).unstack().loc[CL+TF])
pc=e9[e9.method=='pooled-context']; pc=pc.assign(thr=pc.n_test/pc.pred_sec)
print('PC TabPFN no-ttl', pc[['fit_sec','pred_sec','thr','n_context']].median())
# TTL
print('== TTL ablation: no-ttl minus all, BA, by model (mean), and by target')
pv=to.pivot_table(index=['target','k','seed','model'],columns='features',values='bacc'); pv['d']=pv['no-ttl']-pv['all']
print(pv.groupby('model').d.agg(['mean','min','max']).round(4).loc[CL+TF])
print(pv.groupby(['target','model']).d.mean().unstack()[CL+TF].round(3))
print(pv.groupby(['k','model']).d.mean().unstack()[CL+TF].round(3))
for m in TF+['RF']:
    x=pv.xs(m,level='model').d; print(m,'wilcoxon p',wilcoxon(x).pvalue)
# Do TFMs still lead without TTL?
N=to[to.features=='no-ttl']; mt=N.groupby(['k','model']).bacc.mean().unstack()[CL+TF].round(3); print('no-ttl BA by k'); print(mt)
# source effect without TTL
tn=N[N.model=='TabPFN'][['target','seed','k','bacc']]
m1=pc.merge(tn,on=['target','seed','k'],suffixes=('','_to')); m1['d']=m1.bacc-m1.bacc_to
e8t=e8[(e8.model=='TabPFN')&(e8.method=='target-only')].drop_duplicates(['target','seed','k'])[['target','seed','k','bacc']]
e8p=e8[(e8.model=='TabPFN')&(e8.method=='pooled-context')&(e8.S==500)&(e8.k.isin([1,10,100]))]
m2=e8p.merge(e8t,on=['target','seed','k'],suffixes=('','_to')); m2['d']=m2.bacc-m2.bacc_to
for nm,m in [('no-ttl',m1),('all',m2)]:
    pp=m.groupby(['source','target']).d.mean()
    b=[rng.choice(pp.values,12).mean() for _ in range(10000)]
    print(nm,'pooled dBA %.4f [%.4f,%.4f]'%(pp.mean(),*np.percentile(b,[2.5,97.5])),'p=%.3f'%wilcoxon(pp).pvalue,
          'by k',m.groupby('k').d.mean().round(4).to_dict(),'by target',m.groupby('target').d.mean().round(3).to_dict())
    mf=m.copy()
# F1 too
for nm,df,base in [('no-ttl',pc,N),('all',e8p,e8[(e8.method=='target-only')])]:
    bt=base[base.model=='TabPFN'].drop_duplicates(['target','seed','k'])[['target','seed','k','f1']]
    mm=df.merge(bt,on=['target','seed','k'],suffixes=('','_to')); mm['d']=mm.f1-mm.f1_to
    pp=mm.groupby(['source','target']).d.mean(); b=[rng.choice(pp.values,12).mean() for _ in range(10000)]
    print(nm,'pooled dF1 %.4f [%.4f,%.4f]'%(pp.mean(),*np.percentile(b,[2.5,97.5])),'p=%.3f'%wilcoxon(pp).pvalue)
