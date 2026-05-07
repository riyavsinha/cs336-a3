import numpy as np
import scipy.optimize

from cs336_scaling.client import save_final_submission
from cs336_scaling.hw.flop_calibration import full_run_flops
from cs336_scaling.hw.utils import calc_tokens, non_embedding_params
from cs336_scaling.training.model.basic_model import BasicTransformerConfig
from cs336_scaling.training.optimizer import AdamWConfig, WarmupCosineDecay
from cs336_scaling.training.training_config import TrainingConfig

RUNTIME_MINS = 48 * 60
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 16
N_EVALS = 16
LR = 4e-3
D_MODEL = 1920
N_LAYERS = 15
C = full_run_flops()

# from running models
RUNS = [
  (5308416, 1.700465e16, 4.204102),
  (12582912, 1.700465e16, 4.127930),
  (24576000, 1.700465e16, 4.297852),
  (67436544, 1.700465e16, 4.832031),
  (100663296, 1.700465e16, 5.103516),
  (12582912, 1.020279e17, 3.770508),
  (24576000, 1.020279e17, 3.757812),
  (67436544, 1.020279e17, 3.881836),
  (100663296, 1.020279e17, 3.988281),
  (196608000, 1.020279e17, 4.270508),
  (24576000, 6.121674e17, 3.521484),
  (67436544, 6.121674e17, 3.460938),
  (100663296, 6.121674e17, 3.503906),
  (196608000, 6.121674e17, 3.593750),
  (339738624, 6.121674e17, 3.711914),
]


def loss_law(x, c, a, alpha, b, beta):
  n, d = x
  return a * n ** (-alpha) + b * d ** (-beta) + c


def predict_loss(N, D):
  ns = np.array([n for n, _, _ in RUNS])
  ds = np.array([c / (6 * n) for n, c, _ in RUNS])
  losses = np.array([loss for _, _, loss in RUNS])
  (c, a, alpha, b, beta), _ = scipy.optimize.curve_fit(
    loss_law,
    (ns, ds),
    losses,
  )
  return loss_law((N, D), c, a, alpha, b, beta)


def make_config() -> TrainingConfig:
  N = non_embedding_params(D_MODEL, N_LAYERS)
  intermediate_size = int(round((8 * D_MODEL / 3) / 64) * 64)
  return TrainingConfig(
    architecture_config=BasicTransformerConfig(
      attention_bias=False,
      head_dim=64,
      hidden_size=D_MODEL,
      intermediate_size=intermediate_size,
      num_attention_heads=D_MODEL // 64,
      num_hidden_layers=N_LAYERS,
      num_key_value_heads=D_MODEL // 64,
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
  config = make_config()
  N = non_embedding_params(D_MODEL, N_LAYERS)
  print(predict_loss(N, config.total_train_tokens))
  # print(save_final_submission(config, predict_loss(N, config.total_train_tokens)))
