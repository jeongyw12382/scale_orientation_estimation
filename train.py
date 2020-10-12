import os, argparse
import yaml
import wandb
import pdb
import tqdm

import torch

from utils.config import CfgNode
from torch.utils.data import DataLoader, random_split

from model.naive_mlp import L2NetNaiveModel
from model.losses import MultiLoss, PeakyLoss, CSELoss
from datautil.cifar import CIFARSelfSupScale

from evaluation import shifting_validation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True
    )
    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)
    else:
        print('Config File Not Exists!')
        exit(0)
        
    if cfg.metadata.wandb_log:
        print('Initializing WANDB!')
        wandb.init(
            project='so_estimation',
            name=cfg.metadata.exp_name,
            config=cfg
        )
    
    assert cfg != None
    model = eval(cfg.model.name)(**cfg.model.kw)
    trainable_parameters = list(model.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.name)(trainable_parameters, **cfg.optimizer.kw)
    
    dataset = eval(cfg.dataset.name)(**cfg.dataset)
    dataset_train, dataset_val = random_split(dataset, [
        int(cfg.metadata.train_val_ratio * len(dataset)),
        len(dataset) - int(cfg.metadata.train_val_ratio * len(dataset))
    ])
    dataloader_train = DataLoader(dataset_train, cfg.metadata.bsz, collate_fn=dataset.collate_fn, shuffle=True)
    dataloader_val = DataLoader(dataset_val, cfg.metadata.bsz, collate_fn=dataset.collate_fn, shuffle=True)
    
    dir_path = os.path.join('logs', cfg.metadata.exp_name)
    if os.path.exists(dir_path):
        print('A log file already exists!')
        exit(0)
    
    iscuda = not -1 in cfg.metadata.gpu
    loss_class = MultiLoss(*cfg.model.loss, **cfg.dataset.scale)
    best_acc, best_epoch, best_model = -1.0, -1, None
    
    if iscuda:
        model = model.cuda()
    for epoch in range(cfg.metadata.epoch):
        print('Running Epoch : {}\n'.format(epoch))
        for data in tqdm.tqdm(dataloader_train):
            feat = model(data, iscuda)
            loss, loss_dict = loss_class(**feat)
            if cfg.metadata.wandb_log:
                wandb.log(loss_dict)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if epoch % cfg.metadata.validate_every == 0:
            acc = shifting_validation(dataloader_val, model, iscuda, cfg)
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_model = model.state_dict()
                print('Best Acc Updated! Acc: {}'.format(best_acc))
            
            if cfg.metadata.wandb_log:
                wandb.log({
                    'validation/acc': acc,
                    'validation/best_acc': best_acc
                })
            
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            ckpt_path = os.path.join(dir_path, 'checkpoint_{}.ckpt'.format(epoch))
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'acc' : acc
            }, ckpt_path)
            
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'acc' : acc
    }, os.path.join(dir_path, 'best_model.ckpt'))
                
    
if __name__=='__main__':
    main()