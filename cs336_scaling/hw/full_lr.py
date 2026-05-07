import pandas as pd

from cs336_scaling.hw.flop_calibration import calc_compute_budget
from cs336_scaling.hw.models import N_EVALS, calc_n_layers
from cs336_scaling.hw.utils import all_done, calc_tokens, get_best_loss, get_last_loss, non_embedding_params, print_exp, run
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

LR_RUNTIME_MINS = 45
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
D_MODEL = 1920
LR_VALUES = [5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 3e-2, 5e-2]
LR_COMPUTE_BUDGET = calc_compute_budget(LR_RUNTIME_MINS)


def make_full_training_config(lr: float) -> TrainingConfig:
  n_layers = calc_n_layers(D_MODEL)
  N = non_embedding_params(D_MODEL, n_layers)
  intermediate_size = int(round((8 * D_MODEL / 3) / 64) * 64)
  return TrainingConfig(
    architecture_config=BasicTransformerConfig(
      attention_bias=False,
      head_dim=64,
      hidden_size=D_MODEL,
      intermediate_size=intermediate_size,
      num_attention_heads=D_MODEL // 64,
      num_hidden_layers=n_layers,
      num_key_value_heads=D_MODEL // 64,
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
    total_train_tokens=calc_tokens(LR_COMPUTE_BUDGET, N, TRAIN_BATCH_SIZE, N_EVALS),
    max_runtime_seconds=LR_RUNTIME_MINS * 60,
    model_seed=0,
  )


if __name__ == "__main__":
  exps = []
  for lr in LR_VALUES:
    config = make_full_training_config(lr)
    exp = run(config)
    exps.append(exp)
    print_exp(exp)

  if all_done(exps):
    df = pd.DataFrame([
      {
        "lr": exp.training_config.optimizer_config.lr_scheduler.peak_value,
        "final_loss": get_last_loss(exp),
      }
      for exp in exps
    ])
    print(df.to_latex(index=False, float_format="%.3e"))
    best = get_best_loss(exps)
    best_lr = best.training_config.optimizer_config.lr_scheduler.peak_value
    print(f"best_lr={best_lr} best_val_loss={get_last_loss(best)}")
