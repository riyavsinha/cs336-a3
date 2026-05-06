from cs336_scaling.hw.flop_calibration import calc_compute_budget
from cs336_scaling.hw.models import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, N_EVALS, calc_n_layers
from cs336_scaling.hw.utils import calc_tokens, eval_progress_str, get_best_loss, get_last_loss, non_embedding_params, non_embedding_params_from_config, run
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

LR_RUNTIME_MINS = 45
D_MODEL = 1920
LR_VALUES = [3e-3, 1e-2, 3e-2, 5e-2]
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
    print(config)
    exp = run(config)
    exps.append(exp)
    N = non_embedding_params_from_config(config)
    print(
      f"lr={lr:.4e} d_model={D_MODEL} N={N} tokens={config.total_train_tokens} "
      f"id={exp.experiment_id} status={exp.status.status_type} evals={eval_progress_str(exp)}"
    )

    if exp.status.status_type in ("completed", "failed"):
      print(f"last_val_loss={get_last_loss(exp):.6f}")

  if all(e.status.status_type in ("completed", "failed") for e in exps):
    best = get_best_loss(exps)
    best_lr = best.training_config.optimizer_config.lr_scheduler.peak_value
    print(f"best_lr={best_lr} best_val_loss={get_last_loss(best)}")
