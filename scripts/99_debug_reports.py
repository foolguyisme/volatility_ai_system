import glob
import json
import os

ps = sorted(glob.glob("artifacts/reports/*.json"))
print("n =", len(ps))
print()

for p in ps[-10:]:
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)

    print("----", os.path.basename(p))
    print(" tag =", j.get("tag"))
    print(" target =", j.get("target"))
    training = j.get("training") or {}
    print(" training.target =", training.get("target"))
    print(" target_col =", j.get("target_col"))
    print(" keys contains:", [k for k in ["target", "target_col", "training", "rmse_mean", "rmse", "tag"] if k in j])
    print()
