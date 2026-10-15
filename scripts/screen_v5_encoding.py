"""Bounded initial screen on the historical development split."""
from pathlib import Path
import sys, json, hashlib, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from src.house_prices_v5_encoding import build_model
from scripts.optimize_v2 import partitions
from scripts.run_experiments import load_data


def main():
    art=ROOT/"experiments/v5/encoding_screen";art.mkdir(exist_ok=True)
    train,_=load_data();dev,_=partitions(train);x=train.iloc[dev];y=np.log(x.SalePrice.to_numpy())
    specs={"te_hist_7":{"smooth":20.,"leaves":7},"te_hist_15":{"smooth":20.,"leaves":15}}
    (art/"specifications.json").write_text(json.dumps(specs,indent=2)+"\n")
    names=["ref_v2_linear","ref_v2_cat","en_inter_size_scaled","cat_d5_onehot"]
    base=sum(np.load(ROOT/"experiments/v3/cv"/f"dev_{n}_seed42.npz")["prediction"]*.25 for n in names)
    masks={"all":np.ones(len(y),bool),"ordinary":~((x.GrLivArea.to_numpy()>4000)&(x.SalePrice.to_numpy()<300000)),"normal":x.SaleCondition.eq("Normal").to_numpy()}
    rows=[]
    for name,spec in specs.items():
        p=np.zeros(len(y));folds=np.full(len(y),-1);start=time.time()
        for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(x)):
            with threadpool_limits(limits=2):
                model=build_model(spec,42+fold).fit(x.iloc[tr],y[tr]);p[va]=model.predict(x.iloc[va])
            folds[va]=fold
            print(name,fold+1,flush=True)
        np.savez_compressed(art/f"{name}_seed42.npz",prediction=p,ids=x.Id.to_numpy(),fold_ids=folds)
        for w in [1.,.15,.30]:
            pred=p*w+base*(1-w)
            rows.append({"model":name,"new_weight":w,"seconds":time.time()-start,**{k:float(np.sqrt(np.mean((pred[m]-y[m])**2))) for k,m in masks.items()}})
    rows.append({"model":"v4","new_weight":0.,**{k:float(np.sqrt(np.mean((base[m]-y[m])**2))) for k,m in masks.items()}})
    results=pd.DataFrame(rows);results.to_csv(art/"results.csv",index=False);print(results.to_string(index=False),flush=True)


if __name__=="__main__":main()
