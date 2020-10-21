import os, argparse
import yaml

import pdb
import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import torch

from utils.config import CfgNode
from torch.utils.data import DataLoader, random_split

from model.naive_mlp import *
from model.losses import *
from datautil.cifar import *

from evaluation import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True
    )
    # parser.add_argument(
    #     '--loda-checkpoint', type=str, action=store_true
    # )
    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg_dict['dataset']['scale']['min_scale'] = 1.0
            cfg_dict['dataset']['scale']['max_scale'] = cfg_dict['dataset']['scale']['scale_factor']
            cfg = CfgNode(cfg_dict)
    else:
        print('Config File Not Exists!')
        exit(0)
        
    now = datetime.now()
    dt_string = now.strftime("_%Y_%m_%d_%H_%M_%S")    
    
    if cfg.metadata.tbd_log:
        print('Initializing TBD!')
        writer = SummaryWriter('runs/{}'.format(cfg.metadata.exp_name + dt_string))
    
    assert cfg != None
    model = eval(cfg.model.name)(**cfg.model)
    trainable_parameters = list(model.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.name)(trainable_parameters, **cfg.optimizer.kw)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.metadata.gpu)
    print('Running on GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    dataset = eval(cfg.dataset.name)(**cfg.dataset)
    torch.manual_seed(0)
    dataset_train, dataset_val = random_split(dataset, [
        int(cfg.metadata.train_val_ratio * len(dataset)),
        len(dataset) - int(cfg.metadata.train_val_ratio * len(dataset))
    ])
    torch.manual_seed(torch.initial_seed())
    dataloader_train = DataLoader(dataset_train, cfg.metadata.bsz, collate_fn=dataset.collate_fn, shuffle=True)
    dataloader_val = DataLoader(dataset_val, cfg.metadata.bsz, collate_fn=dataset.collate_fn, shuffle=True)
    
    dir_path = os.path.join('logs', cfg.metadata.exp_name + dt_string)
    if os.path.exists(dir_path):
        print('A log file already exists!')
        exit(0)
    
    iscuda = -1 != cfg.metadata.gpu
    loss_class = MultiLoss(*cfg.model.loss, **cfg.dataset.scale)
    best_acc, best_epoch, best_model = -1.0, -1, None
    step = 0
    val_loss = []
    
    print('Running Experiment {}'.format(cfg.metadata.exp_name + dt_string))
    
    if iscuda:
        model = model.cuda()
    for epoch in range(cfg.metadata.epoch):
        print('Running Epoch : {}\n'.format(epoch))
        
        model.train()
        
        for data in tqdm.tqdm(dataloader_train):
            feat = model(data, iscuda)
            train_kw = {**feat, **cfg, 'step': step}
            if cfg.metadata.tbd_log: train_kw['writer'] = writer
            loss = loss_class(**train_kw)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
        
        if epoch % cfg.metadata.validate_every == 0:
            model.eval()
            val_kw = {
                'dataloader': dataloader_val, 
                'model': model,
                'iscuda': iscuda,
                'cfg': cfg,
                'step': epoch // cfg.metadata.validate_every
            }
            if cfg.metadata.tbd_log: val_kw['writer'] = writer
            acc = eval(cfg.validation.name)(**val_kw)
            val_loss.append(acc)
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_model = model.state_dict()
                print('Best Acc Updated! Acc: {}'.format(best_acc))
            
            if cfg.metadata.tbd_log:
                log_dict = {'validation/acc': acc, 'validation/best_acc': best_acc}
            
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
                
            ckpt_path = os.path.join(dir_path, 'checkpoint_{}.ckpt'.format(epoch))
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'acc' : acc
            }, ckpt_path)
            
            if cfg.metadata.early_stop:
                thr = cfg.metadata.early_stop_thr
                if epoch < cfg.metadata.early_stop_thr:
                    continue
                condition = [val_loss[epoch-i-1] < val_loss[epoch-i] for i in range(thr)]
                if True in condition: continue
                else: break
            
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'acc' : acc
    }, os.path.join(dir_path, 'best_model.ckpt'))
                
    
if __name__=='__main__':
    main()