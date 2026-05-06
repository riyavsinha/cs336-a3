from cs336_scaling.schemas import ExperimentResponse
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig
from cs336_scaling.training.training_config import TrainingConfig
from cs336_scaling.hw.utils import run

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
  cfg = exp.training_config
  n = 12 * cfg.architecture_config.num_hidden_layers * cfg.architecture_config.hidden_size**2
  d = cfg.total_train_tokens
  t = exp.status.used_runtime_seconds
  return 6.0 * n * d / t

if __name__ == "__main__":
  exp = run(SMALL_CONFIG)
  print(f"experiment_id={exp.experiment_id}")
  print(f"status={exp.status.status_type}")
  if exp.status.status_type == "completed":
    eff = flops(exp)
    print(f"flops/sec={eff:.6e}")
    print(f"48h flops={48 * 3600 * eff}")