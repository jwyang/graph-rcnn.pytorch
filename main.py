"""
Implementation of ECCV 2018 paper "Graph R-CNN for Scene Graph Generation".
Author: Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, Devi Parikh
Contact: jw2yang@gatech.edu
"""

import os
import pprint
import argparse
import numpy as np
import torch
import datetime

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a scene graph generation model')
    parser.add_argument('--model', dest='model', help='options: grcnn, imp, msdn, nmotif', default='grcnn', type=str)
    parser.add_argument('--backbone', dest='backbone', help='options: vgg16, res50, res101, res152', default='vgg16', type=str)
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vg_bm', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval', help='number of iterations to display', default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="server", nargs=argparse.REMAINDER)
    parser.add_argument('--nw', dest='nworker', help='number of workers', default=0, type=int)
    parser.add_argument('--cuda', dest='cuda', help='whether use cuda', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple gpus for training', action='store_true')
    parser.add_argument('--pretrain', dest='pretrain', help='whether it is pretraining faster r-cnn', action='store_true')
    # config optimization
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parser.add_argument('--lr', dest='lr_base', help='base learning rate', default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='step to do learning rate decay, unit is epoch', default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session', help='training session', default=1, type=int)

    # training mode
    parser.add_argument('--mode', dest='mode', help='training mode, 0:scratch, 1:resume or 2:finetune', default=0, type=int)
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model', default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard', help='whether use tensorflow tensorboard', default=False, type=bool)

    # parser.add_argument('--imdb_name', dest='imdb_name', help='imdb to train on', default='imdb_512.h5', type=str)
    # parser.add_argument('--imdb', dest='imdb', help='imdb to train on', default='imdb_512.h5', type=str)
    # parser.add_argument('--roidb', dest='roidb', default='VG', type=str)
    # parser.add_argument('--rpndb', dest='rpndb', default='proposals.h5', type=str)

    args = parser.parse_args()
    return args

def train(cfg, args):
    """
    train scene graph generation model
    """
    arguments = {}
    arguments["iteration"] = 0
    model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.train()
    return model

def test(cfg, args, model=None):
    """
    test scene graph generation model
    """
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.test(visualize=args.visualize)

def main():
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Graph Reasoning Machine for Visual Question Answering")
    parser.add_argument("--config-file", default="configs/baseline_res101.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=-1)
    parser.add_argument("--use_freq_prior", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--algorithm", type=str, default='sg_baseline')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.resume = args.resume
    cfg.instance = args.instance
    cfg.inference = args.inference
    cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
    cfg.MODEL.ALGORITHM = args.algorithm
    # cfg.freeze()

    if not os.path.exists("logs") and get_rank() == 0:
        os.mkdir("logs")
    logger = setup_logger("scene_graph_generation", "logs", get_rank(),
        filename="{}_{}.txt".format(args.algorithm, get_timestamp()))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    output_config_path = os.path.join("logs", 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    if not args.inference:
        model = train(cfg, args)
    else:
        test(cfg, args)

if __name__ == "__main__":
    main()
