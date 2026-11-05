import sys, numpy as np, pandas as pd
R='/'.join(__file__.split('/')[:-2])
BASE=sys.argv[1]
sys.path.insert(0, BASE+'/replication_v32/pipeline')
import _base as B
B.DATA=BASE+'/wrds_data'
fu=B.screen_full_universe()
meme=set(B.screen()['permno'].astype(int))
q=fu[(fu.car_pct>50)&(fu.car_pct<=100)&(~fu.permno.isin(meme))]['permno'].astype(int).tolist()
print('benchmark n',len(q))
pm=B.build_matches(q, exclude=meme)
print('matched',len(pm))
pan=B.bhar_panel(pm, horizons=[48,58])
s=pd.read_csv(BASE+'/v13_outputs/contemp_placebo_bhar48.csv')
h=pan[pan.horizon_mo==48].merge(s,on='permno',suffixes=('_new','_saved'),how='outer')
h['diff']=(h.bhar_new-h.bhar_saved).abs()
print('rows',len(h),'max abs diff',h['diff'].max(),'n exact(<1e-9)',(h['diff']<1e-9).sum(),'missing new',h.bhar_new.isna().sum(),'missing saved',h.bhar_saved.isna().sum())
print(h[h['diff']>1e-9].head(20).to_string())
out=pd.DataFrame([{'permno':k,'ctrl_permno':v} for k,v in pm.items()])
out.to_csv(sys.argv[2]+'/bench_pairs.csv',index=False); pan.to_csv(sys.argv[2]+'/bench_bhar.csv',index=False)
