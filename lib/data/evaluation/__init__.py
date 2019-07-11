from .coco import coco_evaluation
from .voc import voc_evaluation
from .sg import sg_evaluation

from ..vg_hdf5 import vg_hdf5

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """

    if isinstance(dataset, vg_hdf5):
        args = dict(
            dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
        )
        # return voc_evaluation(**args)
        return coco_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))

def evaluate_sg(dataset, predictions, predictions_pred, output_folder, **kwargs):
    """evaluate scene graph generation performance
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        predictions_pred(list[BoxList]): each item in the list represents the
            predicate prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, predictions_pred=predictions_pred, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, vg_hdf5):
        return sg_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
