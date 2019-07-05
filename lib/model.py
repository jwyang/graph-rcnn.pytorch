import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger

class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        self.data_loader_train = build_data_loader(cfg, split="train", is_distributed=distributed)
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser)

        self.arguments.update(self.extra_checkpoint_data)

        if distributed:
            self.scene_parser = torch.nn.parallel.DistributedDataParallel(
                self.scene_parser, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        for i, data in enumerate(self.data_loader_train, start_iter):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            loss_dict = self.scene_parser(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            self.sp_optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if i % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 and i > start_iter:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if i == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def test(self):
        """
        main body for testing scene graph generation model
        """

def build_model(cfg, arguments, local_rank, distributed):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed)
