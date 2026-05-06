from cs336_scaling.hw.flop_calibration import SMALL_CONFIG, calc_compute_budget
from cs336_scaling.hw.utils import calc_tokens, eval_progress_str, get_best_loss, get_last_loss, non_embedding_params_from_config, run
from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

RUNTIME_MINS = 12
C1 = calc_compute_budget(RUNTIME_MINS)
LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2]

def make_lr_config(lr, total_train_tokens, max_runtime_seconds):
  cfg_arch = SMALL_CONFIG.architecture_config
  return TrainingConfig(
    architecture_config=cfg_arch,
    optimizer_config=AdamWConfig(
      lr_scheduler=WarmupCosineDecay(peak_value=lr),
    ),
    train_batch_size=128,
    val_batch_size=32,
    n_evals=16,
    total_train_tokens=total_train_tokens,
    max_runtime_seconds=max_runtime_seconds,
    model_seed=0,
  )


def get_best_lr(exps: list[ExperimentResponse]):
  best = get_best_loss(exps)
  lr = best.training_config.optimizer_config.lr_scheduler.peak_value
  loss = get_last_loss(best)
  return lr, loss


def lr_scale_from_reference(best_lr, C, *, C_ref=C1):
  return best_lr * (C / C_ref) ** (-0.125)


if __name__ == "__main__":
  N = non_embedding_params_from_config(SMALL_CONFIG)
  d_tokens = calc_tokens(C1, N)

  exps = [run(make_lr_config(lr, d_tokens, RUNTIME_MINS * 60)) for lr in LR_VALUES]

  for lr, exp in zip(LR_VALUES, exps):
    print(
      f"lr={lr} experiment_id={exp.experiment_id} status={exp.status.status_type} evals={eval_progress_str(exp)}"
    )

  if not all(e.status.status_type in ("completed", "failed") for e in exps):
    print("Waiting.")
  else:
    best_lr, best_loss = get_best_lr(exps)
    print(f"best_lr={best_lr} best_val_loss={best_loss}")
  
