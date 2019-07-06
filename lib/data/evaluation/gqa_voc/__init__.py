import logging

from .gqa_voc_eval import do_gqa_voc_evaluation


def gqa_voc_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("graph_reasoning_machine.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    return do_gqa_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
