import cv2
import torch

def select_top_predictions(predictions, confidence_threshold=0.2):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score
    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def compute_colors_for_labels(labels, palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions, categories):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image

def overlay_question_answers(image, qas, max_num=10):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    height, width = image.shape[0], image.shape[1]
    template = "Question: {} \t Answer: {}"
    for i, qa in enumerate(qas):
        x, y = 20, (height - 20 - 20 * i)
        s = template.format(qa[0], qa[1])
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1
        )
    return image
