import sys, pandas as pd, numpy as np
BASE=sys.argv[1]; OUT=sys.argv[2]
sys.path.insert(0, BASE+'/replication_v32/pipeline')
import _base as B
B.DATA=BASE+'/wrds_data'
fc=B.full_ccm().set_index('permno'); fu=B.screen_full_universe().set_index('permno')
R=B._returns_by_permno()
def last_trade(p):
    r=R.get(int(p)); return None if r is None else r['date'].max().date().isoformat()
def table(pairs, bhar):
    rows=[]
    for sp,cp in pairs.items():
        s=fc.loc[sp]; c=fc.loc[cp]; u=fu.loc[sp]
        basis='book-to-market' if pd.notna(s.bm_ratio) and pd.notna(c.bm_ratio) else 'size'
        row={'permno':sp,'ticker':u.ticker,'company':u.comnam,'crsp_siccd':u.siccd,'compustat_sic':s.sic,
             'event_car_pct_compounded':u.car_pct,'mktcap_m_dec2020':s.mktcap_m,'book_to_market':s.bm_ratio,
             'ctrl_permno':cp,'ctrl_ticker':c.ticker,'ctrl_company':c.comnam,'ctrl_compustat_sic':c.sic,
             'ctrl_mktcap_m_dec2020':c.mktcap_m,'ctrl_book_to_market':c.bm_ratio,
             'matched_sic1':str(s.sic)[0],'size_ratio':c.mktcap_m/s.mktcap_m,
             'bm_distance':abs(c.bm_ratio-s.bm_ratio) if basis=='book-to-market' else np.nan,
             'match_basis':basis,'event_firm_last_return':last_trade(sp),'ctrl_last_return':last_trade(cp)}
        for hz in (48,58):
            x=bhar[(bhar.permno==sp)&(bhar.horizon_mo==hz)]
            row[f'bhar_{hz}m']=x.bhar.iloc[0] if len(x) else np.nan
            row[f'event_bhr_{hz}m']=x.meme_bhr.iloc[0] if len(x) else np.nan
            row[f'ctrl_bhr_{hz}m']=x.ctrl_bhr.iloc[0] if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('event_car_pct_compounded',ascending=False)
mp=pd.read_csv(OUT+'/meme_pairs.csv'); bp=pd.read_csv(OUT+'/bench_pairs.csv')
d1=table(dict(zip(mp.permno,mp.ctrl_permno)), pd.read_csv(OUT+'/meme_bhar.csv'))
d2=table(dict(zip(bp.permno,bp.ctrl_permno)), pd.read_csv(OUT+'/bench_bhar.csv'))
d1.to_csv(OUT+'/D1.csv',index=False); d2.to_csv(OUT+'/D2.csv',index=False)
fu.reset_index()[['permno','ticker','company' if 'company' in fu else 'comnam','siccd','exchcd','car_pct','n_est_days','n_event_days']].to_csv(OUT+'/D4.csv',index=False)
print(len(d1),len(d2),len(fu))
print('D1 bhar48 mean',round(d1.bhar_48m.mean()*100,2),'median',round(d1.bhar_48m.median()*100,2))
print('D2 bhar48 mean',round(d2.bhar_48m.mean()*100,2),'median',round(d2.bhar_48m.median()*100,2))
print('basis counts D1',d1.match_basis.value_counts().to_dict(),'D2',d2.match_basis.value_counts().to_dict())
print('D1 ctrl ending before 2025-02-04:',(d1.ctrl_last_return<'2025-02-04').sum(),' D2:',(d2.ctrl_last_return<'2025-02-04').sum())
print('D4 >100:',(fu.car_pct>100).sum(),' 50-100 non-meme:',((fu.car_pct>50)&(fu.car_pct<=100)).sum())
print('sic1 agree D1',(d1.compustat_sic.astype(str).str[0]==d1.ctrl_compustat_sic.astype(str).str[0]).all(),'D2',(d2.compustat_sic.astype(str).str[0]==d2.ctrl_compustat_sic.astype(str).str[0]).all())
print('size band ok',d1.size_ratio.between(0.3,2.0).all(),d2.size_ratio.between(0.3,2.0).all())
