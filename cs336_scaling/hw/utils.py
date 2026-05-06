import json
import math
from pathlib import Path

from cs336_scaling.client import get_experiment, submit_experiment
from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.training_config import TrainingConfig

EXP_RECORD_PATH = Path("experiment_id_map.json")
SEQ_LEN = 512

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
  
def non_embedding_params(config: TrainingConfig):
  return 12 * config.architecture_config.num_hidden_layers * config.architecture_config.hidden_size**2

def calc_tokens(C, N, batch_size=128, n_evals=16):
  d = int(math.ceil(C / (6 * N)))
  m = SEQ_LEN * batch_size * n_evals
  return ((d + m - 1) // m) * m
