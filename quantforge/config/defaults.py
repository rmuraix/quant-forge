"""Default config templates for `quantforge init`."""

BASIC_TEMPLATE = """\
version: 1

project:
  name: resnet18-cifar10
  output_dir: outputs
  seed: 42

model:
  name: resnet18
  pretrained: true
  num_classes: null
  in_chans: 3
  checkpoint_path: null

dataset:
  name: cifar10
  subset: null
  split:
    train: train
    validation: test
    test: null
  image_column: img
  label_column: label
  cache_dir: null

preprocess:
  image_size: 224
  mean: null
  std: null
  interpolation: bicubic

augment:
  random_resized_crop: true
  horizontal_flip: true
  randaugment: false
  mixup: 0.0
  cutmix: 0.0

training:
  epochs: 10
  batch_size: 64
  lr: 0.0003
  weight_decay: 0.05
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 1
  amp: true
  grad_clip_norm: null
  num_workers: 4
  device: auto
  deterministic: false

quantization:
  enabled: false
  backend: none
  mode: none
  dtype: int8
  recipe: null
  calibration_samples: 512

checkpoint:
  resume: null
  save_best: true
  monitor: accuracy

tracking:
  backend: wandb
  project: quantforge
  entity: null
  run_name: null
  group: null
  job_type: null
  tags: []
  notes: null
  mode: online
  log_config: true
  log_metrics: true
  log_system: true
  log_predictions: false
  max_prediction_samples: 32
  watch_model: false
  upload_artifacts: false
  upload_checkpoints: false
  upload_quantized_models: false

logging:
  level: info
  log_interval: 50
  rich: true
"""

QAT_TEMPLATE = """\
version: 1

project:
  name: resnet18-cifar10-qat
  output_dir: outputs
  seed: 42

model:
  name: resnet18
  pretrained: true
  num_classes: null
  in_chans: 3
  checkpoint_path: null

dataset:
  name: cifar10
  subset: null
  split:
    train: train
    validation: test
    test: null
  image_column: img
  label_column: label
  cache_dir: null

preprocess:
  image_size: 224
  mean: null
  std: null
  interpolation: bicubic

augment:
  random_resized_crop: true
  horizontal_flip: true
  randaugment: false
  mixup: 0.0
  cutmix: 0.0

training:
  epochs: 10
  batch_size: 64
  lr: 0.0001
  weight_decay: 0.05
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 1
  amp: false
  grad_clip_norm: null
  num_workers: 4
  device: auto
  deterministic: false

quantization:
  enabled: true
  backend: torchao
  mode: qat
  dtype: int8
  recipe: null
  calibration_samples: 512

checkpoint:
  resume: null
  save_best: true
  monitor: accuracy

tracking:
  backend: wandb
  project: quantforge
  entity: null
  run_name: null
  group: null
  job_type: qat
  tags: [qat]
  notes: null
  mode: online
  log_config: true
  log_metrics: true
  log_system: true
  log_predictions: false
  max_prediction_samples: 32
  watch_model: false
  upload_artifacts: false
  upload_checkpoints: false
  upload_quantized_models: false

logging:
  level: info
  log_interval: 50
  rich: true
"""

PTQ_TEMPLATE = """\
version: 1

project:
  name: resnet18-cifar10-ptq
  output_dir: outputs
  seed: 42

model:
  name: resnet18
  pretrained: true
  num_classes: null
  in_chans: 3
  checkpoint_path: null

dataset:
  name: cifar10
  subset: null
  split:
    train: train
    validation: test
    test: null
  image_column: img
  label_column: label
  cache_dir: null

preprocess:
  image_size: 224
  mean: null
  std: null
  interpolation: bicubic

augment:
  random_resized_crop: true
  horizontal_flip: true
  randaugment: false
  mixup: 0.0
  cutmix: 0.0

training:
  epochs: 10
  batch_size: 64
  lr: 0.0003
  weight_decay: 0.05
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 1
  amp: true
  grad_clip_norm: null
  num_workers: 4
  device: auto
  deterministic: false

quantization:
  enabled: true
  backend: torchao
  mode: ptq
  dtype: int8
  recipe: null
  calibration_samples: 512

checkpoint:
  resume: null
  save_best: true
  monitor: accuracy

tracking:
  backend: wandb
  project: quantforge
  entity: null
  run_name: null
  group: null
  job_type: ptq
  tags: [ptq]
  notes: null
  mode: online
  log_config: true
  log_metrics: true
  log_system: true
  log_predictions: false
  max_prediction_samples: 32
  watch_model: false
  upload_artifacts: false
  upload_checkpoints: false
  upload_quantized_models: false

logging:
  level: info
  log_interval: 50
  rich: true
"""

TEMPLATES = {
    "basic": BASIC_TEMPLATE,
    "qat": QAT_TEMPLATE,
    "ptq": PTQ_TEMPLATE,
}
