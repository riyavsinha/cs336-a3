import datetime as dt
import math

import pandas as pd

from cs336_scaling.hw.flop_calibration import calc_compute_budget, full_run_flops
from cs336_scaling.hw.isoflops_laws import fit_data, fit_params, plot_against_compute, plot_loss_curves
from cs336_scaling.hw.utils import calc_tokens, eval_progress_str, get_last_loss, non_embedding_params, non_embedding_params_from_config, run
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 32
N_EVALS = 16

T1_MINS = 3.3
C_LEVELS = [calc_compute_budget(T1_MINS), calc_compute_budget(T1_MINS * 6), calc_compute_budget(T1_MINS * 36)]
RUNTIME_SECS = [T1_MINS * 60, T1_MINS * 6 * 60, T1_MINS * 36 * 60]

DM_BY_LEVEL = [
  [384, 512, 640, 896, 1024],
  [512, 640, 896, 1024, 1280],
  [640, 896, 1024, 1280, 1536],
]


def calc_n_layers(d_model):
  return round(d_model / 128)

def make_config(d_model, C, max_runtime_seconds):
  n_layers = calc_n_layers(d_model)
  N = non_embedding_params(d_model, n_layers)
  lr = 3.2e-3 - 1.4e-4 * math.log10(N)
  intermediate_size = int(round((8 * d_model / 3) / 64) * 64)
  return TrainingConfig(
    architecture_config=BasicTransformerConfig(
      attention_bias=False,
      head_dim=64,
      hidden_size=d_model,
      intermediate_size=intermediate_size,
      num_attention_heads=d_model // 64,
      num_hidden_layers=n_layers,
      num_key_value_heads=d_model // 64,
      rms_norm_eps=1e-6,
      rope_theta=1_000_000,
      tie_word_embeddings=False,
      dtype="bfloat16",
      vocab_size=32_000,
    ),
    optimizer_config=AdamWConfig(
      lr_scheduler=WarmupCosineDecay(peak_value=lr),
    ),
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    n_evals=N_EVALS,
    total_train_tokens=calc_tokens(C, N, TRAIN_BATCH_SIZE, N_EVALS),
    max_runtime_seconds=max_runtime_seconds,
    model_seed=0,
  )


def pred_cs(df, n_intervals=7):
  lo = float(df["compute_budget"].min())
  target = float(full_run_flops())
  inv = 1 / n_intervals
  hi = target * (target / lo) ** inv
  a, b = math.log(lo), math.log(hi)
  pts = [math.exp(a + (b - a) * i * inv) for i in range(n_intervals + 1)]
  return sorted({*pts, target})


if __name__ == "__main__":
  rows = []
  for i, (C, dms, runtime) in enumerate(zip(C_LEVELS, DM_BY_LEVEL, RUNTIME_SECS), start=1):
    for d_model in dms:
      config = make_config(d_model, C, runtime)
      exp = run(config)
      N = non_embedding_params_from_config(config)
      lr = config.optimizer_config.lr_scheduler.peak_value
      if exp.status.status_type == "completed":
        true_rt = exp.status.used_runtime_seconds
      elif exp.status.status_type == "failed":
        true_rt = runtime
      elif exp.status.status_type == "running":
        true_rt = (dt.datetime.now(dt.timezone.utc) - exp.status.dispatched_at).total_seconds()
      else:
        true_rt = 0
      print(
        f"level=C{i} C={C:.3e} d_model={d_model} N={N} lr={lr:.4e} "
        f"tokens={config.total_train_tokens} runtime={true_rt:.1f}s(expected: {config.max_runtime_seconds:.1f}s) "
        f"id={exp.experiment_id} status={exp.status.status_type} evals={eval_progress_str(exp)}"
      )

      if exp.status.status_type in ("completed", "failed"):
        final_loss = get_last_loss(exp)
        print(f"last_val_loss={final_loss:.6f}")
      else:
        final_loss = float("nan")
      rows.append({"parameters": N, "compute_budget": C, "final_loss": final_loss})

  df = pd.DataFrame(rows)
  print(df)

  fit_df = df.dropna(subset=["final_loss"])
  plot_loss_curves(fit_df, "figs/3_loss.png")

  cbs = pred_cs(fit_df)
  min_df, fit_n = fit_params(fit_df)
  preds_n = pd.DataFrame({"compute_budget": cbs, "parameters": [fit_n(c) for c in cbs]})
  print(pd.concat([min_df, preds_n]))
  plot_against_compute(pd.concat([min_df, preds_n]), "parameters", "Parameters", "figs/3a_params.png")

  min_d_df, fit_d = fit_data(fit_df)
  preds_d = pd.DataFrame({"compute_budget": cbs, "d": [fit_d(c) for c in cbs]})
  print(pd.concat([min_d_df, preds_d]))
  plot_against_compute(pd.concat([min_d_df, preds_d]), "d", "Data Size", "figs/3b_data.png")
