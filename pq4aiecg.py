# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import models.encoder as encoder
import util.misc as misc
from engine_downstream import evaluate, train_one_epoch
from util.dataset import build_dataset, get_dataloader
from util.losses import build_loss_fn
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from util.perf_metrics import build_metric_fn, is_best_metric


def parse() -> dict:
    parser = argparse.ArgumentParser('ECG downstream training')

    parser.add_argument('--config_path',
                        # required=True,
                        type=str,
                        default='./configs/downstream/st_mem.yaml',
                        metavar='FILE',
                        help='YAML config file path')
    parser.add_argument('--output_dir',
                        default='downstream_out',
                        type=str,
                        metavar='DIR',
                        help='path where to save')
    parser.add_argument('--exp_name',
                        default="pq_dowstream",
                        type=str,
                        help='experiment name')
    parser.add_argument('--resume',
                        default="",
                        type=str,
                        metavar='PATH',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument('--encoder_path',
                        default="./models/encoder/st_mem_vit_base.pth",
                        type=str,
                        metavar='PATH',
                        help='pretrained encoder checkpoint')

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def main(config):
    misc.init_distributed_mode(config['ddp'])

    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    # fix the seed for reproducibility
    seed = config['seed'] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = False

    # ECG dataset
    dataset_train = build_dataset(config['dataset'], split='train')
    dataset_valid = build_dataset(config['dataset'], split='valid')

    data_loader_train = get_dataloader(dataset_train,
                                       is_distributed=config['ddp']['distributed'],
                                       mode='train',
                                       **config['dataloader'])
    data_loader_valid = get_dataloader(dataset_valid,
                                       is_distributed=config['ddp']['distributed'],
                                       dist_eval=config['train']['dist_eval'],
                                       mode='eval',
                                       **config['dataloader'])

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None

    model_name = config['model_name']
    print(f"Model name: {model_name}")##Model name: st_mem_vit_base
    if model_name in encoder.__dict__:
        model = encoder.__dict__[model_name](**config['model'])
        
    else:
        raise ValueError(f'Unsupported model name: {model_name}')

    if config['mode'] != "scratch":
        # print(f"Load encoder from: {config['encoder_path']}")## ./models/encoder/st_mem_vit_base.pth
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        # print('checkpoint.keys()',checkpoint.keys())##['epoch', 'model', 'optimizer', 'scaler', 'config'])
        # print("checkpoint['epoch']",checkpoint['epoch'])##799
        # print("checkpoint['optimizer']",checkpoint['optimizer'])##None
        # print("checkpoint['scaler']",checkpoint['scaler'])##None
        # print("checkpoint['config']",checkpoint['config'])
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:##
            '''
            这个循环检查 'head.weight' 和 'head.bias' 这两个键。
            如果它们在预训练模型中存在，且形状与当前模型不匹配，就从预训练模型中删除这些键。
            这通常是因为头部（输出层）在不同任务间可能有所不同。
            '''
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        ###the following code is used to check the model's parameters are loaded correctly
        # for name, param in model.named_parameters():
        #     if name in checkpoint_model:
        #         print(f"{name}: {torch.allclose(param, checkpoint_model[name])}")
        #         break
        # print('----------the following is msg----------------------------')
        input = torch.randn(2, 12, 2250)
        output = model(input)
        print('output',output.shape)


if __name__ == '__main__':
    config = parse()
    main(config)
