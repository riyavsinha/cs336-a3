import pandas as pd

from cs336_scaling.hw.models import N_EVALS
from cs336_scaling.hw.utils import all_done, calc_tokens, flops_per_sec, get_best_loss, get_last_loss, non_embedding_params, print_exp, run
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

RUNTIME_MINS = 45
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
LR = 4e-3
FLOPS_PER_SEC = 5.2e14 # from fps from models2
C = RUNTIME_MINS * 60 * FLOPS_PER_SEC

SHAPES = [
  (2048, 18),
  (2176, 16),
  (2240, 15),
]


def make_config(d_model: int, n_layers: int) -> TrainingConfig:
  N = non_embedding_params(d_model, n_layers)
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
      lr_scheduler=WarmupCosineDecay(peak_value=LR),
    ),
    train_batch_size=TRAIN_BATCH_SIZE,
    val_batch_size=VAL_BATCH_SIZE,
    n_evals=N_EVALS,
    total_train_tokens=calc_tokens(C, N, TRAIN_BATCH_SIZE, N_EVALS),
    max_runtime_seconds=RUNTIME_MINS * 60,
    model_seed=0,
  )


if __name__ == "__main__":
  exps = []
  for d_model, n_layers in SHAPES:
    exp = run(make_config(d_model, n_layers))
    exps.append(exp)
    print_exp(exp)

  if all_done(exps):
    df = pd.DataFrame([
      {
        "d_model": exp.training_config.architecture_config.hidden_size,
        "n_layers": exp.training_config.architecture_config.num_hidden_layers,
        "parameters": non_embedding_params(exp.training_config.architecture_config.hidden_size, exp.training_config.architecture_config.num_hidden_layers),
        "final_loss": get_last_loss(exp),
        "flops_per_sec": flops_per_sec(exp),
      }
      for exp in exps
    ])
    print(df.to_latex(index=False, float_format="%.3e"))
    best = get_best_loss(exps)
    config = best.training_config
    print(f"best_d_model={config.architecture_config.hidden_size} best_n_layers={config.architecture_config.num_hidden_layers} best_val_loss={get_last_loss(best)}")
