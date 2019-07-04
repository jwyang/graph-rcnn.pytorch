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

from lib.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.datasets.factory import build_dataloader

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

def train(model):
    """
    train scene graph generation model
    """

def test(model):
    """
    test scene graph generation model
    """
    
def main():
    """
    code for config
    """

    ''' parse config file '''
    args = parse_args()
    args.cfg_file = "configs/{}_{}.yml".format(args.model, args.backbone)
    cfg_from_file(args.cfg_file)
    print('Using config:')
    pprint.pprint(cfg)
    print('Setting random seed: ', cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    ''' build dataloader '''
    dataloader = build_dataloader(dataset="vg_bm")

    ''' build model and optimizer '''


    ''' train model '''


if __name__ == "__main__":
    main()
