import json, subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor
imgs=["astronaut","chelsea","coffee","rocket","ihc","retina"]
M=["pe_single","pe","ipe_gauss","ipe_box","ipe_hyb","pe_ss","ipe_hyb_c","ipe_gauss_c"]
J=[]
for i in imgs:
  for m in M: J.append(dict(image=i,method=m,seed=0,tag="main",L=10,ts="2,4,8,16",pred=True))
for i in imgs:
  for m in ["pe","ipe_gauss","ipe_box","ipe_hyb","pe_ss","ipe_hyb_c","ipe_gauss_c"]: J.append(dict(image=i,method=m,seed=0,tag="sparse",L=10,ts="2,16"))
for i in imgs:
  for m in M: J.append(dict(image=i,method=m,seed=1,tag="main",L=10,ts="2,4,8,16"))
for i in ["astronaut","coffee"]:
  for L in [6,8,12,16]:
    for m in ["pe","ipe_gauss","ipe_box"]: J.append(dict(image=i,method=m,seed=0,tag="Lsweep",L=L,ts="2,4,8,16"))
def key(d): return (d["image"],d["method"],d["seed"],d["tag"],d["L"])
done=set()
if os.path.exists("results.jsonl"):
  for l in open("results.jsonl"):
    r=json.loads(l); done.add((r["image"],r["method"],r["seed"],r["tag"],r["L"]))
skip={("astronaut","ipe_hyb_c",0,"main",10),("astronaut","ipe_gauss_c",0,"main",10)} if os.environ.get("SKIPRUN") else set()
todo=[j for j in J if key(j) not in done and key(j) not in skip]
print(len(J),"total",len(todo),"todo",flush=True)
def go(j):
  cmd=["python3","fipe.py","--image",j["image"],"--method",j["method"],"--seed",str(j["seed"]),"--tag",j["tag"],"--L",str(j["L"]),"--train_scales",j["ts"]]
  if j.get("pred"): cmd+=["--save_pred",f"preds/{j['image']}_{j['method']}.npz"]
  subprocess.run(cmd,env={**os.environ,"NT":"1"},stdout=subprocess.DEVNULL,stderr=open("logs/err.log","a"))
with ThreadPoolExecutor(2) as ex: list(ex.map(go,todo))
print("ALL DONE",flush=True)
