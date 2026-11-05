import sys, pandas as pd
BASE=sys.argv[1]; sys.path.insert(0, BASE+'/replication_v32/pipeline')
import _base as B
B.DATA=BASE+'/wrds_data'
meme=sorted(set(B.screen()['permno'].astype(int)))
mm=B.build_matches(meme)
a=pd.read_csv(BASE+'/../../AppendixD_matched_control_assignments.csv')
d=dict(zip(a.permno,a.ctrl_permno))
print('n',len(mm),'agree w AppendixD',sum(d.get(k)==v for k,v in mm.items()))
pan=B.bhar_panel(mm,horizons=[48,58])
s=pd.read_csv(BASE+'/wrds_data/analysis_bhar_matched.csv')
for hz in (48,58):
    x=pan[pan.horizon_mo==hz].merge(s[s.horizon_mo==hz],on='permno')
    print(hz,len(x),'max diff',(x.bhar_x-x.bhar_y).abs().max(), 'mean',round(x.bhar_x.mean()*100,2))
pd.DataFrame([{'permno':k,'ctrl_permno':v} for k,v in mm.items()]).to_csv(sys.argv[2]+'/meme_pairs.csv',index=False)
pan.to_csv(sys.argv[2]+'/meme_bhar.csv',index=False)
