from __future__ import print_function

import random
import yaml
import os
from argparse import ArgumentParser
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vit_pytorch import ViT
from optimizers import make_optimizer

def seed_everything(seed):
    if seed is None: return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(config):
    I = config.model['image_size']
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((I,I), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((I,I)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # test_transforms = transforms.Compose([
    #     # transforms.Resize(256),
    #     # transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    # ])
    root = config.train['data_root']
    train_data = datasets.CIFAR10(root, train=True, transform=train_transform)#, download=True)
    test_data = datasets.CIFAR10(root, train=False, transform=test_transform)
    # test_data = datasets.CIFAR10(root, train=False, transform=test_transform)

    batch_size = config.train['batch_size']
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
    @property
    def avg(self):
        return self.sum / self.cnt if self.cnt!=0 else None

class Accuracy:
    def __init__(self):
        self.sum = 0.
        self.right = 0
    def update(self, right, batch_size):
        self.sum += batch_size
        self.right += right
    @property
    def val(self):
        return self.right/self.sum if self.sum!=0 else 0

def test():
    model = ViT(
        image_size= 32,
        patch_size= 32,
        num_classes= 10,
        channels= 3,
        dim= 128,
        depth= 6,
        heads= 16,
        mlp_dim= 2048,
        dropout= 0.1,
        emb_dropout= 0.1
    )
    data = datasets.CIFAR10('./cifar10', train=True)
    x = torch.randn((30, 3, 32, 32), dtype=torch.float32)
    print(model(x).shape)

def main():
    # args
    GB = 1024**3
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--project', type=str, default='adamw_decomposed_experiment')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--p', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.name != '':
        args.name = '_' + args.name
    with open(f'./config/{args.config}.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if args.device is not None:
            config['train']['device'] = args.device
        wandb.init(
            project=args.project, 
            config=config,
            name=config['train']['optim']+args.name
        )
        config = wandb.config
    # settings
    device = config.train['device']
    seed_everything(args.seed or config.train['seed'])
    # lr = config.train['lr']
    # input(f'{lr}, {type(lr)}')
    factor = config.train['factor']
    optimizer_choice = config.train['optim']
    epochs = config.train['epochs']
    # data
    train_loader, valid_loader = load_data(config)
    # model
    model = ViT(**config.model).to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    if args.p is not None and config.optimizer.get('p') is None:
        config.optimizer['p'] = args.p
    optimizer = make_optimizer(
        optimizer_choice, model,
        model.parameters(), **config.optimizer
    )
    tqdm.write(f'{type(optimizer)}')
    tqdm.write(f'config: {config.optimizer}')
    # scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        patience=2, 
        factor=factor,
        mode='min'
    )
    # train
    torch.cuda.reset_peak_memory_stats(device)
    # wandb.watch(model, log='parameters', log_freq=100)
    for epoch in range(epochs):
        epoch_loss = AverageMeter()
        epoch_accuracy = Accuracy()

        with tqdm(
            train_loader, 
            desc=f'epoch[{epoch+1}]'
        ) as process_bar:
            model.train()
            for idx, (data, label) in enumerate(process_bar):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                # wandb.log({
                #     'mem-alloc-backward': torch.cuda.memory_allocated(device)/GB,
                # })
                optimizer.step()

                right = torch.sum(output.argmax(dim=1) == label).item()
                epoch_accuracy.update(right, label.shape[0])
                epoch_loss.update(loss.item())
                if idx%70==0:
                    wandb.log({
                        'train-loss': epoch_loss.avg,
                        'train-acc': 100 * epoch_accuracy.val,
                        'mem-alloc': torch.cuda.memory_allocated(device)/GB,
                    })
                    process_bar.set_postfix(
                        train_loss = f'{epoch_loss.avg :.4f}', 
                        train_acc = f'{epoch_accuracy.val * 100 :.2f}%'
                    )

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = Accuracy()
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                right = torch.sum(val_output.argmax(dim=1) == label).item()
                epoch_val_accuracy.update(right, label.shape[0])
                epoch_val_loss += val_loss / len(valid_loader)
            wandb.log({
                'val-loss': epoch_val_loss,
                'val-acc': epoch_val_accuracy.val * 100
            })
        scheduler.step(epoch_val_loss, epoch=epoch)

        tqdm.write(
            f"Epoch : {epoch+1} - loss : {epoch_loss.avg:.4f} - acc: {epoch_accuracy.val:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy.val:.4f}\n"
        )
    model_save_path = config.train['model_save']
    torch.save(model.state_dict(), model_save_path)
    wandb.save(model_save_path)
    tqdm.write(f'Saved in {model_save_path}')


if __name__=='__main__':
    main()
    # test()