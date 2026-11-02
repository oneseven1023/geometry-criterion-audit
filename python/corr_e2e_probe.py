import os
import numpy as np
import pandas as pd

rows = []


def run(path, tag):
    df = pd.read_csv(path)
    y = df["aoa_sz"].to_numpy(float)
    m = np.isfinite(y)
    out = {"tag": tag, "n": int(m.sum())}
    for c, name in [("kappa", "kappa"), ("d_eff", "d_eff"),
                    ("share_z", "share_z"), ("mean_abs_uz", "mean|u_z|")]:
        x = df[c].to_numpy(float)
        mm = m & np.isfinite(x)
        out[name] = float(np.corrcoef(x[mm], y[mm])[0, 1])
    rows.append(out)


run("results/e2e_links.csv", "base(sd0.10,th5)")
for tag, d in [("sd=0.05", "e2e_sens_sd005"), ("sd=0.20", "e2e_sens_sd020"),
               ("th=2.5deg", "e2e_sens_th025"), ("th=10deg", "e2e_sens_th10")]:
    p = "results/%s/e2e_links.csv" % d
    if os.path.exists(p):
        run(p, tag)

t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.to_string(index=False, float_format=lambda v: "%.3f" % v))
