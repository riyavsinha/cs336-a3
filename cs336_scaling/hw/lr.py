from cs336_scaling.hw.flop_calibration import SMALL_CONFIG, calc_runtime
from cs336_scaling.hw.utils import calc_tokens, non_embedding_params, run
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

C1 = 6e18
LR_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

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


def best_completed_lr(exps):
  done = [e for e in exps if e.status.status_type == "completed"]
  if len(done) != len(exps):
    raise ValueError("not all LR experiments are completed")

  best = min(done, key=lambda e: e.status.val_losses[-1])
  best_lr = best.training_config.optimizer_config.lr_scheduler.peak_value
  best_loss = best.status.val_losses[-1]
  return best_lr, best_loss


def lr_scale_from_reference(best_lr, C, *, C_ref=C1):
  return best_lr * (C / C_ref) ** (-0.125)


if __name__ == "__main__":
  N = non_embedding_params(SMALL_CONFIG)
  d_tokens = calc_tokens(C1, N)
  max_runtime_seconds = calc_runtime(C1)

  exps = [run(make_lr_config(lr, d_tokens, max_runtime_seconds)) for lr in LR_VALUES]

  for lr, exp in zip(LR_VALUES, exps):
    print(f"lr={lr} experiment_id={exp.experiment_id} status={exp.status.status_type}")

  if all(e.status.status_type == "completed" for e in exps):
    best_lr, best_loss = best_completed_lr(exps)
    print(f"best_lr={best_lr} best_val_loss={best_loss}")
