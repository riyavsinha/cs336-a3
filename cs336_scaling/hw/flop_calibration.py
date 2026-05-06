from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig
from cs336_scaling.hw.utils import non_embedding_params, run

SMALL_CONFIG = TrainingConfig(
  architecture_config=BasicTransformerConfig(
    attention_bias=False,
    head_dim=64,
    hidden_size=384,
    intermediate_size=1536,
    num_attention_heads=6,
    num_hidden_layers=11,
    num_key_value_heads=6,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000,
    tie_word_embeddings=False,
    dtype="bfloat16",
    vocab_size=32_000,
  ),
  optimizer_config=AdamWConfig(),
  train_batch_size=128,
  val_batch_size=32,
  n_evals=16,
  total_train_tokens=10 * 1_048_576,  # 10M
  max_runtime_seconds=120.0,
  model_seed=0,
)

def flops(exp: ExperimentResponse) -> float:
  n = non_embedding_params(exp.training_config)
  d = exp.training_config.total_train_tokens
  t = exp.status.used_runtime_seconds
  return 6.0 * n * d / t

def calc_runtime(C):
  return max(1.0, 1.5 * C / flops_per_sec())


def calc_compute_budget(mins):
  return 60.0 * mins * flops_per_sec()


def flops_per_sec():
  exp = run(SMALL_CONFIG)
  if exp.status.status_type == "completed":
    return flops(exp)
  raise Exception("wait")

if __name__ == "__main__":
  fps = flops_per_sec()
  print(f"flops/sec={fps:.6e}")
  print(f"48h flops={48 * 3600 * fps}")