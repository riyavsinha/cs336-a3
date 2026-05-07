import datetime as dt
import json
import math
from pathlib import Path

from cs336_scaling.client import get_experiment, submit_experiment
from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.training_config import TrainingConfig

EXP_RECORD_PATH = Path("experiment_id_map.json")
SEQ_LEN = 512

# Final constants
FULL_PARAMS = 6.803062e+08
FULL_DM = 1.260586e+10

def load_map() -> dict[str, int]:
  if not EXP_RECORD_PATH.exists():
    return {}
  with EXP_RECORD_PATH.open() as f:
    data = json.load(f)
  return {str(k): int(v) for k, v in data.items()}


def save_map(mapping: dict[str, int]) -> None:
  with EXP_RECORD_PATH.open("w") as f:
    json.dump(mapping, f, indent=2, sort_keys=True)


def run(config: TrainingConfig) -> ExperimentResponse:
  """Submit config or return the existing experiment for same config."""
  uid = config.unique_id
  mapping = load_map()
  cached_id = mapping.get(uid)
  if cached_id is not None:
    return get_experiment(cached_id)
  submitted = submit_experiment(config)
  exp = get_experiment(submitted.experiment_id)
  mapping[uid] = exp.experiment_id
  save_map(mapping)
  return exp
  
def non_embedding_params_from_config(config: TrainingConfig):
  return non_embedding_params(config.architecture_config.hidden_size, config.architecture_config.num_hidden_layers)

def non_embedding_params(d_model, n_layers):
  return 12 * n_layers * d_model**2

def calc_tokens(C, N, batch_size=128, n_evals=16):
  d = int(math.ceil(C / (6 * N)))
  m = SEQ_LEN * batch_size * n_evals
  return ((d + m - 1) // m) * m

def flops_per_sec(exp: ExperimentResponse):
  N = non_embedding_params_from_config(exp.training_config)
  d = exp.training_config.total_train_tokens
  return 6 * N * d / exp.status.used_runtime_seconds

def get_last_loss(exp: ExperimentResponse):
  st = exp.status
  if st.status_type == "completed":
    return st.val_losses[-1]
  if st.status_type == "failed" and st.reason.reason == "timeout" and st.reason.partial_val_losses:
    return st.reason.partial_val_losses[-1]
  return math.inf

def eval_progress(exp: ExperimentResponse):
  total = exp.training_config.n_evals
  st = exp.status
  if st.status_type == "queued":
    done = 0
  elif st.status_type in ("running", "completed"):
    done = len(st.val_losses)
  elif st.status_type == "failed" and st.reason.reason == "timeout":
    done = len(st.reason.partial_val_losses)
  else:
    done = 0
  return done, total


def eval_progress_str(exp: ExperimentResponse):
  done, total = eval_progress(exp)
  return f"{done}/{total}"


def print_exp(exp: ExperimentResponse):
  config = exp.training_config
  N = non_embedding_params_from_config(config)
  lr = config.optimizer_config.lr_scheduler.peak_value
  C = 6 * N * config.total_train_tokens
  if exp.status.status_type == "completed":
    runtime = exp.status.used_runtime_seconds
  elif exp.status.status_type == "failed":
    runtime = getattr(exp.status, "used_runtime_seconds", config.max_runtime_seconds)
  elif exp.status.status_type == "running":
    runtime = (dt.datetime.now(dt.timezone.utc) - exp.status.dispatched_at).total_seconds()
  else:
    runtime = 0
  print(f"C={C:.3e} d_model={config.architecture_config.hidden_size} N={N} lr={lr:.4e} tokens={config.total_train_tokens} runtime={runtime:.1f}s(expected: {config.max_runtime_seconds:.1f}s) id={exp.experiment_id} status={exp.status.status_type} evals={eval_progress_str(exp)}")
  if exp.status.status_type in ("completed", "failed"):
    print(f"last_val_loss={get_last_loss(exp):.6f}")


def get_best_loss(exps: list[ExperimentResponse]):
  assert all(e.status.status_type in ("completed", "failed") for e in exps), "Not all experiments done yet"
  return min(exps, key=get_last_loss)

def all_done(exps: list[ExperimentResponse]):
  return all(e.status.status_type in ("completed", "failed") for e in exps)