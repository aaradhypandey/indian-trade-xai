from __future__ import annotations
import argparse, csv, json, math, os, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

PARSER=argparse.ArgumentParser(description='PSO-LIME-SHAP monthly Indian trade analysis')
PROJECT_ROOT = Path(__file__).resolve().parent
PARSER.add_argument('--data', type=Path, default=PROJECT_ROOT/'TradeDataUpdated.xlsx')
PARSER.add_argument('--out', type=Path, default=PROJECT_ROOT)
ARGS=PARSER.parse_args()
DATA=ARGS.data
OUT=ARGS.out
RESULTS=OUT/'results'; FIG=OUT/'figures'
RESULTS.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)

TASKS=[
 ('Crude Oil','CrudeOil','Import_Qty','International demand (imports)'),
 ('Cotton','Cotton','Import_Qty','International demand (imports)'),
 ('Electronic Circuits','Circuits','Import_Qty','International demand (imports)'),
 ('Rice','Rice','Export_Qty','International supply (exports)'),
 ('Packed Medicine','PackedMedicine','Export_Qty','International supply (exports)'),
 ('Spices','Spices','Export_Qty','International supply (exports)'),
]

# ---------- Spreadsheet reader (stdlib only; workbook itself is not edited here) ----------
def read_xlsx(path:Path):
    zf=zipfile.ZipFile(path)
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    ss=[]
    if 'xl/sharedStrings.xml' in zf.namelist():
        root=ET.fromstring(zf.read('xl/sharedStrings.xml'))
        for si in root.findall('m:si',ns):
            ss.append(''.join(t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
    wb=ET.fromstring(zf.read('xl/workbook.xml')); rel=ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    relmap={x.attrib['Id']:x.attrib['Target'] for x in rel}
    sheets={}
    for s in wb.find('m:sheets',ns):
        name=s.attrib['name']; rid=s.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']; tgt=relmap[rid]
        root=ET.fromstring(zf.read('xl/'+tgt if not tgt.startswith('xl/') else tgt))
        rows=[]
        for row in root.findall('.//m:sheetData/m:row',ns):
            d={}
            for c in row.findall('m:c',ns):
                ref=c.attrib['r']; col=''.join(ch for ch in ref if ch.isalpha())
                t=c.attrib.get('t'); v=c.find('m:v',ns); val=None if v is None else v.text
                if t=='s' and val is not None: val=ss[int(val)]
                d[col]=val
            rows.append(d)
        sheets[name]=rows
    return sheets

sheets=read_xlsx(DATA)

def get_rows(sheet_name):
    raw=sheets[sheet_name]
    headers=[]
    # first row contains headers A:O
    for i in range(15): headers.append(raw[0].get(chr(65+i),''))
    out=[]
    for r in raw[1:]:
        try:
            year=int(float(r.get('A'))); month=int(float(r.get('B')))
            if year in range(2021,2026) and month in range(1,13):
                vals=[float(r.get(chr(65+i))) for i in range(15)]
                out.append(dict(zip(headers,vals)))
        except Exception:
            continue
    assert len(out)==60, (sheet_name,len(out))
    return out

def events(y):
    y=np.asarray(y,float)
    ch=np.zeros(len(y)); ch[1:]=(y[1:]-y[:-1])/y[:-1]
    z=np.zeros(len(y))
    usable=ch[1:]
    z[1:]=(usable-usable.mean())/usable.std(ddof=1)
    lab=np.array(['REGULAR']*len(y),dtype=object)
    lab[z>=1.5]='SURGE'; lab[z<=-1.5]='DROP'
    return ch,z,lab

def pso_fit(X,y,seed,n_particles=80,n_iter=250):
    """Continuous full-vector PSO. A particle is one feature-weight vector."""
    rng=np.random.default_rng(seed); n,d=X.shape
    pos=rng.uniform(-1.5,1.5,(n_particles,d)); vel=rng.uniform(-0.2,0.2,(n_particles,d))
    def nonlinear(W):
        # feature-weighted nonlinear additive response
        out=np.empty((n,W.shape[0]))
        for j,w in enumerate(W): out[:,j]=np.sum(w*(0.2*X+0.8*np.tanh(X)),axis=1)
        return out
    def fitness(W):
        pred=nonlinear(W); return np.mean((pred-y[:,None])**2,axis=0)+1e-4*np.sum(W**2,axis=1)
    pbest=pos.copy(); pfit=fitness(pos)
    k=int(np.argmin(pfit)); gbest=pbest[k].copy(); gfit=float(pfit[k]); hist=[]
    for it in range(n_iter):
        r1=rng.random((n_particles,d)); r2=rng.random((n_particles,d))
        inertia=0.9-0.4*it/max(1,n_iter-1)
        vel=inertia*vel+1.6*r1*(pbest-pos)+1.6*r2*(gbest[None,:]-pos)
        pos=np.clip(pos+vel,-2,2)
        f=fitness(pos); better=f<pfit
        pbest[better]=pos[better]; pfit[better]=f[better]
        k=int(np.argmin(pfit))
        if pfit[k]<gfit: gbest=pbest[k].copy(); gfit=float(pfit[k])
        hist.append(gfit)
    return gbest,np.array(hist)

def gfun(x): return 0.2*x+0.8*np.tanh(x)

def lime_explain(x0,w,rng,n_samples=3000):
    d=len(w); Z=x0+rng.normal(0,0.45,(n_samples,d)); Z[0]=x0
    pred=np.sum(w*(0.2*Z+0.8*np.tanh(Z)),axis=1)
    dist=np.sum((Z-x0)**2,axis=1); kw=np.exp(-dist/(d*0.8**2))
    model=Ridge(alpha=1e-5,fit_intercept=True)
    model.fit(Z,pred,sample_weight=kw)
    return model.coef_

all_month=[]; event_detail=[]; pso_all=[]; shap_all=[]; summaries=[]
for si,(sector,sheet,target,role) in enumerate(TASKS):
    rows=get_rows(sheet); y=np.array([r[target] for r in rows],float)
    value_col='Import_Value_USD' if target=='Import_Qty' else 'Export_Value_USD'
    predictors=[h for h in rows[0].keys() if h not in ['Year','Month',target,value_col]]
    X=np.array([[r[h] for h in predictors] for r in rows],float)
    tr=np.arange(48)
    mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd==0]=1
    ym=y[tr].mean(); ys=y[tr].std(); ys=ys or 1
    Xstd=(X-mu)/sd; ystd=(y-ym)/ys
    w,hist=pso_fit(Xstd[tr],ystd[tr],200+si)
    pshare=np.abs(w)/(np.abs(w).sum()+1e-12)*100
    # yearly SHAP for additive model: phi_i(x)=w_i*(g(x_i)-E[g(X_i)])
    bg=np.mean(gfun(Xstd[tr]),axis=0)
    phi=w[None,:]*(gfun(Xstd)-bg[None,:])
    rng=np.random.default_rng(1000+si)
    outlier_count=0; diffs=[]
    top_counts={}
    for i,r in enumerate(rows):
        c=lime_explain(Xstd[i],w,rng)
        lime_local=np.abs(c*Xstd[i])  # local contribution from the fitted LIME surrogate
        ps_local=np.abs(w*gfun(Xstd[i]))  # local contribution induced by final gbest weights
        lshare=lime_local/(lime_local.sum()+1e-12)*100
        ps_local_share=ps_local/(ps_local.sum()+1e-12)*100
        top=int(np.argmax(lshare)); diff=float(abs(lshare[top]-ps_local_share[top]))
        event=events(y)[2][i]
        ch,z,lab=events(y)
        out='YES' if diff>10 else 'NO'; outlier_count+=out=='YES'; diffs.append(diff)
        top_counts[predictors[top]]=top_counts.get(predictors[top],0)+1
        all_month.append([sector,int(r['Year']),int(r['Month']),f"{int(r['Year'])}-{int(r['Month']):02d}",event,float(ch[i]*100),float(z[i]),predictors[top],float(lshare[top]),float(ps_local_share[top]),diff,out])
        if event in ('SURGE','DROP'):
            event_detail.append([sector,int(r['Year']),int(r['Month']),event,float(ch[i]*100),float(z[i]),predictors[top],float(lshare[top]),float(ps_local_share[top]),diff,out])
    for j,f in enumerate(predictors): pso_all.append([sector,target,f,float(pshare[j]),float(w[j])])
    for year in range(2021,2026):
        mask=np.array([int(r['Year'])==year for r in rows])
        vals=np.mean(np.abs(phi[mask]),axis=0)
        shares=vals/(vals.sum()+1e-12)*100
        for f,val in zip(predictors,shares): shap_all.append([sector,year,f,float(val)])
    ch,z,lab=events(y)
    summaries.append([sector,target,role,int(np.sum(lab=='SURGE')),int(np.sum(lab=='DROP')),int(np.sum(lab=='REGULAR')),float(np.mean(diffs)),float(np.mean(np.array(diffs)>10)*100),predictors[int(np.argmax(pshare))],float(np.max(pshare))])
    np.savetxt(RESULTS/f'pso_convergence_{sheet}.csv',hist,delimiter=',',header='best_objective',comments='')

# output CSVs
def wcsv(path,header,rows):
    with open(path,'w',newline='') as f: csv.writer(f).writerows([header,*rows])
wcsv(RESULTS/'monthly_event_lime_pso.csv',['Sector','Year','Month','YYYY-MM','Event','Target_MoM_Change_pct','Z_score','LIME_top_feature','LIME_local_contribution_pct','PSO_gbest_local_contribution_pct','Difference_pct_points','Outlier_gt10pct'],all_month)
wcsv(RESULTS/'event_month_lime_pso.csv',['Sector','Year','Month','Event','Target_MoM_Change_pct','Z_score','LIME_top_feature','LIME_local_contribution_pct','PSO_gbest_local_contribution_pct','Difference_pct_points','Outlier_gt10pct'],event_detail)
wcsv(RESULTS/'pso_gbest_weights.csv',['Sector','Target','Feature','PSO_gbest_normalized_pct','PSO_raw_weight'],pso_all)
wcsv(RESULTS/'shap_yearly.csv',['Sector','Year','Feature','SHAP_normalized_importance_pct'],shap_all)
wcsv(RESULTS/'task_summary.csv',['Sector','Target','Role','Surges','Drops','Regular','Mean_LIME_PSO_Difference_pct_points','Outlier_rate_pct','Top_PSO_feature','Top_PSO_weight_pct'],summaries)
json.dump({'sector_month_observations':360,'surges':sum(x[4]=='SURGE' for x in all_month),'drops':sum(x[4]=='DROP' for x in all_month),'regular':sum(x[4]=='REGULAR' for x in all_month),'outliers_gt10':sum(x[11]=='YES' for x in all_month),'outlier_rate_pct':100*sum(x[11]=='YES' for x in all_month)/len(all_month)},open(RESULTS/'summary.json','w'),indent=2)
print(open(RESULTS/'summary.json').read())
print('SUMMARIES')
for s in summaries: print(s)
print('EVENT rows',len(event_detail),'Outliers among all months',sum(x[11]=='YES' for x in all_month))
