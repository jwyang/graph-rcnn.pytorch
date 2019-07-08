import logging

from .sg_eval import do_sg_evaluation


def sg_evaluation(dataset, predictions, predictions_pred, output_folder, box_only, **_):
    logger = logging.getLogger("scene_graph_generation.inference")
    logger.info("performing scene graph evaluation.")
    return do_sg_evaluation(
        dataset=dataset,
        predictions=predictions,
        predictions_pred=predictions_pred,
        output_folder=output_folder,
        logger=logger,
    )
