import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from torch.optim import AdamW, lr_scheduler
from configs.basic_cfg import get_cfg
from models.DualVAE import dualvae
import wandb
from utils.metrics import clustering_by_representation
from collections import defaultdict
from utils.datatool import (get_val_transformations,
                            get_train_dataset,
                            get_val_dataset,
                            get_mask_val,
                            add_sp_noise)
from utils.misc import reproducibility_setting
from utils.misc import get_alpha_scheduler
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_device(args, local_rank):
    device = torch.device(f"cuda:{args.train.devices[0]}") if torch.cuda.is_available(
    ) else torch.device('cpu')
    return device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args



def get_scheduler(args, optimizer):
    """
    Optimize learning rate
    """
    if args.train.scheduler == 'constant':
        return None
    elif args.train.scheduler == 'linear':
        lf = lambda x: (1 - x / args.train.epochs) * (1.0 - 0.1) + 0.1  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return scheduler
    elif args.train.scheduler == 'consine':
        eta_min = args.train.lr * (args.train.lr_decay_rate ** 3)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.train.epochs // 10, eta_min=eta_min)
    else:
        scheduler = None
    return scheduler



if __name__ == '__main__':
    args = parse_args()
    config = get_cfg(args.config_file)

    use_wandb = config.wandb
    seed = config.seed
    result_dir = os.path.join(config.train.log_dir,
                              f'{config.experiment_name}-dual-v{config.vspecific.v_dim}-c{config.consistency.c_dim}-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio}-{seed}')
    os.makedirs(result_dir, exist_ok=True)


    device = get_device(config, LOCAL_RANK)
    print(f"Use: {device}")

    seed = config.seed
    reproducibility_setting(seed)

    # Load data
    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(config, val_transformations)



    train_loader = DataLoader(dataset=train_dataset,
                              sampler=None,
                              shuffle=True,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              drop_last=True)

    # Load model
    model = dualvae(
        config=config,
        device=device
    )

    if use_wandb:
        wandb.init(project=config.project_name,
                config=config,
                name=f'{config.experiment_name}-c{config.consistency.c_dim}--v{config.vspecific.v_dim}-m{config.train.masked_ratio}-mv{config.train.mask_view_ratio if config.train.mask_view else 0.0}-{seed}')

    summary(model)
    print('model loaded!')

    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=0.0001, betas=[0.9, 0.95])
    scheduler = get_scheduler(config, optimizer)
    model = model.to(device)


    best_loss = np.inf
    old_best_model_path = ""

    for epoch in range(config.train.epochs):
        lr = optimizer.param_groups[0]['lr']
        print("lr:" + str(lr))
        # Train
        model.train()

        cur_loss = defaultdict(list)
        for Xs, _ in tqdm(train_loader):
            Xs = [x.to(device) for x in Xs]

            loss, details = model.get_loss(Xs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k, v in details.items():
                cur_loss[k].append(v)

        show_losses = {k: np.mean(v) for k, v in cur_loss.items()}

        batch_loss = show_losses['total_loss']
        if use_wandb:
            wandb.log(show_losses, step=epoch)
        print(f"[Epoch {epoch}] | Train loss:{batch_loss.item()}")
        for k, v in show_losses.items():
            print(f"{k}:{v}")

        # Save best model
        if batch_loss <= best_loss:
            best_loss = batch_loss
            best_model_path = os.path.join(result_dir, f"best-{int(batch_loss.item())}-{epoch}-{seed}.pth")
            torch.save(model.state_dict(), best_model_path)
            if old_best_model_path:
                os.remove(old_best_model_path)
            old_best_model_path = best_model_path

        if scheduler is not None:
            scheduler.step()

    final_model_path = os.path.join(result_dir, f"final_model-{seed}.pth")
    torch.save(model.state_dict(), final_model_path)