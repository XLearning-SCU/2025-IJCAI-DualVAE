
import os

from yacs.config import CfgNode as CN


# Basic config, including basic information, dataset, and training setting.
_C = CN()

# project name, for wandb's records. **Required**
_C.project_name = 'CVPR24'
# project description, what problem does this project tackle?
_C.project_desc = 'cvpr new try.'
# seed
_C.seed = 3407
# print log
_C.verbose = True
# enable wandb:
_C.wandb = True
# runtime
_C.runtimes = 1
# experiment name.
_C.experiment_name = "None"
# For multi-view setting
_C.views = 2
# Experiment Notes.
_C.note = ""


# Network setting.
_C.backbone = CN()
_C.backbone.type = "cnn"
# normalizations ['batch', 'layer']
_C.backbone.normalization = 'batch'
# default 'kaiming'.
_C.backbone.init_method = 'kaiming'


# For dataset
_C.dataset = CN()
# ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST',
# 'EdgeMnist', 'FashionMnist', 'coil-20', 'coil-100', 'DHA23', "UWA30"]
_C.dataset.name = 'EdgeMnist'
# rootdir
_C.dataset.root = './data'
# class_num
_C.dataset.class_num = 10



# For augmentation
# training augmentation
_C.training_augmentation = CN()
_C.training_augmentation.enable = True
_C.training_augmentation.crop_size = 32
# Need training augmentation? such as mnist we suggest set as false.
_C.training_augmentation.hflip = True
# random resize crop:
_C.training_augmentation.random_resized_crop = CN()
_C.training_augmentation.random_resized_crop.size = 32
_C.training_augmentation.random_resized_crop.scale = [0.2, 1.0]
# color jitter random apply:
_C.training_augmentation.color_jitter_random_apply = CN()
_C.training_augmentation.color_jitter_random_apply.p = 0.8
# color jitter
_C.training_augmentation.color_jitter = CN()
_C.training_augmentation.color_jitter.brightness = 0.4
_C.training_augmentation.color_jitter.contrast = 0.4
_C.training_augmentation.color_jitter.saturation = 0.4
_C.training_augmentation.color_jitter.hue = 0.1
# random_grayscale
_C.training_augmentation.random_grayscale = CN()
_C.training_augmentation.random_grayscale.p = 0.2


# validation augmentation
_C.valid_augmentation = CN()
# center crop size
_C.valid_augmentation.crop_size = 32



# for training.
_C.train = CN()
_C.train.epochs = 100
_C.train.batch_size = 512
_C.train.optim = 'sgd'
_C.train.devices = [0, 1, 2, 3]
_C.train.lr = 0.001
_C.train.dropout = 0.0
_C.train.num_workers = 2
_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = ""
# the interval of evaluate epoch, defaults to 5.
_C.train.evaluate = 5
# Learning rate scheduler, [cosine, step]
_C.train.scheduler = 'cosine'
_C.train.lr_decay_rate = 0.1
_C.train.lr_decay_epochs = 30
# samling num.
_C.train.samples_num = 6
# using checkpoint training.
_C.train.resume = False
_C.train.ckpt_path = ""
_C.train.use_ddp = True
_C.train.masked_ratio = 0.6
_C.train.mask_patch_size = 2
_C.train.mask_view = True
_C.train.mask_view_ratio = 0.30



# disentagle
_C.disent = CN()
_C.disent.max_mi = False

#specific
_C.vspecific = CN()

# evaluation.
_C.eval = CN()
_C.eval.model_path = './mrdd-weights/emnist.pth'
_C.eval.noise_prob = 0.5
_C.eval.modal_missing_ratio = 0.5
_C.eval.mv_root = './MaskView'
_C.set_new_allowed(is_new_allowed=True)


def get_cfg(config_file_path):
    """
    Initialize configuration.
    """
    config = _C.clone()
    # merge specific config.
    config.merge_from_file(config_file_path)
    if not config.train.log_dir:
        path = f'./experiments/{config.experiment_name}'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    else:
        os.makedirs(config.train.log_dir, exist_ok=True)
    return config