import json
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(fp):
  with open(fp) as f:
    return pd.DataFrame(json.load(f))
  

def plot_loss_curves(df: pd.DataFrame, save_path: str):
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.figure()
  for cb, group in df.groupby("compute_budget"):
    group = group.sort_values("parameters")
    plt.plot(
      group["parameters"],
      group["final_loss"],
      marker="o",
      label=f"{cb:.0e}"
    )
  plt.xlabel("Parameters")
  plt.ylabel("Loss")
  plt.xscale("log")
  plt.title("Loss vs. Parameters")
  plt.grid(True)
  plt.savefig(save_path)
  
def plot_params(df: pd.DataFrame, save_path: str):
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.figure()
  plt.plot(
    df["compute_budget"],
    df["parameters"],
    marker="o",
  )
  plt.xlabel("Compute Budget")
  plt.ylabel("Parameters")
  plt.xscale("log")
  plt.title("Parameters vs Compute Budget")
  plt.grid(True)
  plt.savefig(save_path)

def power_law(C, a, b):
  return a * C ** b
  
def fit(df: pd.DataFrame):
  min_inds = df.groupby("compute_budget").idxmin()["final_loss"]
  df = df.loc[min_inds][["compute_budget", "parameters"]]
  (a, b), c = scipy.optimize.curve_fit(power_law, df["compute_budget"], df["parameters"])
  return df, lambda C: power_law(C, a, b)
    
    
if __name__ == "__main__":
  p = "data/isoflops_curves.json"
  df = load_data(p)
  plot_loss_curves(df, "figs/2a_loss.png")
  min_df, fit_fn = fit(df)
  
  # a)
  cbs = [6e21, 1e22, 3e22, 6e22, 1e23, 3e23, 6e23, 1e24]
  preds = pd.DataFrame({"compute_budget": cbs, "parameters": [fit_fn(cb) for cb in cbs]})
  pred_df = pd.concat([min_df, preds])
  print(preds.to_latex(index=False, float_format="%.3e"))
  plot_params(pred_df, "figs/2a_params.png")
  
  