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
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

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

        if not os.path.exists("freq_prior.npy"):
            freq_prior = self._get_freq_prior()
            np.save("freq_prior.npy", freq_prior)
        else:
            freq_prior = np.load("freq_prior.npy")

        self.freq_prior = freq_prior

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self):
        """
        get the frequency prior for object-pair v.s. predicate
        """
        freq_prior = np.zeros((self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                              self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                              self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES))

        import pdb; pdb.set_trace()
        for i in range(len(self.data_loader_train.dataset)):
            target = self.data_loader_train.dataset.get_groundtruth(i)
            boxes = target.bbox
            overlaps = bbox_overlaps(boxes, boxes)
            labels = target.get_field("labels")
            pred_labels = target.get_field("pred_labels")
            for m in range(pred_labels.size(0)):
                for n in range(pred_labels.size(1)):
                    if pred_labels[m, n] > 0:
                        label_m = labels[m].item()
                        label_n = labels[n].item()
                        freq_prior[label_m, label_n][int(pred_labels[m, n].item())] += 1
                    else:
                        if overlaps[m, n] > 0 and m != n:
                            freq_prior[label_m, label_n][0] += 1
            if i % 20 == 0:
                print("processing {}/{}".format(i, len(self.data_loader_train.dataset)))
            if i >= len(self.data_loader_train.dataset):
                break
        return freq_prior

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
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("maskrcnn_benchmark.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions):
        visualize_folder = "visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            img = imgs.tensors[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_ids[i])), result)

    def test(self, timer=None, visualize=False):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output)
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )
            if i > 1000:
                break
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)
        if self.cfg.MODEL.RELATION_ON:
            predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        if not is_main_process():
            return

        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            if self.cfg.MODEL.RELATION_ON:
                torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))

        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)

        if self.cfg.MODEL.RELATION_ON:
            eval_sg_results = evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            **extra_args)

def build_model(cfg, arguments, local_rank, distributed):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed)
