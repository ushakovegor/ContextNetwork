import warnings
from bidict import bidict
import numpy as np
from endoanalysis.similarity import SimilarityMeasure
def targets_mapping(
    similarity, sim_thresh, targets_gt, targets_pred, scores, multiclass=True
):
    """
    Maps ground truth targets to predictions for a single class.

    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_thresh: list float
        threshold for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates
    targets_gt: endoanalysis.targets.ImageTargets
        ground truth targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    targets_pred: endoanalysis.targets.ImageTargets
        prediction targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    scores: npdarray
        scores (confidences) of the predictions. Must be the same length as targets_pred.
        The predictions with higher score have the priority in mapping
    multiclass : bool
        If True, only the targets with coinsiding class labels will be matched.


    Returns
    -------
    gt_to_pred: bidict.bidict
        mapping from ground truth targets to predictions
    gt_is_tp: ndarray of bool
        the array with true positive labels from ground truth targets
    pred_is_tp: ndarray of bool
        the array with true positive labels from ground pred     targets

    See also
    --------
    https://www.kaggle.com/chenyc15/mean-average-precision-metric
    """

    if len(targets_pred) != len(scores):
        raise ValueError("targets_pred and scores should have the same length")

    preds_ids_sorted = np.argsort(scores)[::-1]
    sim_matrix = similarity.matrix(targets_gt, targets_pred)
    gt_is_tp = np.zeros((len(targets_gt)), dtype=bool)
    pred_is_tp = np.zeros((len(targets_pred)), dtype=bool)
    gt_to_pred = bidict()

    # preds_ids_sorted = np.argsort(scores)
    # sim_matrix = similarity.matrix(targets_gt, targets_pred)

    if multiclass:
        classes_pred = targets_pred.classes()
        classes_gt = targets_gt.classes()
    else:
        classes_pred = np.zeros(len(targets_pred), dtype=int)
        classes_gt = np.zeros(len(targets_gt), dtype=int)

    for pred_j in preds_ids_sorted:
        pred_matched = False

        sims_for_pred = sim_matrix[:, pred_j]
        gt_ids_sorted = np.argsort(sims_for_pred)

        for gt_i in gt_ids_sorted:
            if (
                sim_matrix[gt_i, pred_j] > sim_thresh
                and not gt_is_tp[gt_i]
                and not pred_matched
                and classes_pred[pred_j] == classes_gt[gt_i]
            ):
                pred_matched = True
                pred_is_tp[pred_j] = True
                gt_is_tp[gt_i] = True
                gt_to_pred[gt_i] = pred_j

    return gt_to_pred, gt_is_tp, pred_is_tp


def targets_mapping_multithresh(
    similarity,
    sim_threshs,
    targets_gt,
    targets_pred,
    scores,
):

    """
    Maps ground truth targets to predictions for multiclass targets whith different thresholds for different classes.

    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_threshs: list of float
        thresholds for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates.
        The number of thresholds should be the same as the number of classes.
    targets_gt: endoanalysis.targets.ImageTargets
        ground truth targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    targets_pred: endoanalysis.targets.ImageTargets
        prediction targets (like keypoints or bboxes) for an image. Must be compatible with similarity.
    scores: npdarray
        scores (confidences) of the predictions. Must be the same length as targets_pred.
        The predictions with higher score have the priority in mapping

    Returns
    -------
    gt_to_pred: bidict.bidict
        mapping from ground truth targets to predictions

    Note
    ----
    The classes are assumed to have labels from 0 to num_classes, which is not passed explicitely.
    len(sim_threshs) should be to the number of classes. sim_threshs[0] is the threshold for the
    class_labels 0, sim_threshs[1] is the one for the class label 1 etc.

    See also
    --------
    https://www.kaggle.com/chenyc15/mean-average-precision-metric
    """

    gt_classes = targets_gt.classes()
    pred_classes = targets_pred.classes()
    gt_to_pred = bidict()
    for class_i, thresh in enumerate(sim_threshs):
        gt_ids_for_class = np.where(gt_classes == class_i)[0]
        pred_ids_for_class = np.where(pred_classes == class_i)[0]

        mapping_for_class, _, _ = targets_mapping(
            similarity,
            thresh,
            targets_gt[gt_ids_for_class],
            targets_pred[pred_ids_for_class],
            scores[pred_ids_for_class],
            multiclass=False,
        )

        for i, gt_id in enumerate(gt_ids_for_class):
            if i in mapping_for_class:
                gt_to_pred[gt_id] = pred_ids_for_class[mapping_for_class[i]]

    return gt_to_pred


def compose_tp_labels(
    similarity,
    sim_thresh,
    images_ids,
    targets_batched_gt,
    targets_batched_pred,
    confidences,
):
    """
    Maps predictions to groud truth targets for a batch of images.


    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_thresh: list float
        threshold for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates
    images_ids: list of int
        ids of images to consider
    targets_batched_gt: endoanalysis.targets.ImageTargetsBatch
        ground truth targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    targets_batched_pred: endoanalysis.targets.ImageTargetsBatch
        prediction targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    confidences: ndarray of float
        confdences of the predcitions.

    Returns
    -------
    confidences: ndarray of float
        concatenated confidences of the targets
    pred_is_tp: ndarray of bool
        concatenated true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.


    Note
    ----
    Only the images with ids form images_ids will be present in the result, with the same order as they
    appear in images_ids.
    """

    pred_is_tp = []

    classes = []
    confidences_to_stack = []
    for image_i in images_ids:

        targets_gt = targets_batched_gt.from_image(image_i)
        targets_pred = targets_batched_pred.from_image(image_i)
        # classes_image = targets_batched_pred.from_image(image_i).classes()
        confidences_image = confidences[targets_batched_pred.image_labels() == image_i]
        confidences_to_stack.append(confidences_image)
        _, _, pred_is_tp_image = targets_mapping(
            similarity,
            sim_thresh,
            targets_gt,
            targets_pred,
            confidences_image,
            multiclass=True,
        )
        
        pred_is_tp.append(pred_is_tp_image)
        # classes.append(classes_image)

    pred_is_tp = np.hstack(pred_is_tp)
    confidences = np.hstack(confidences_to_stack)
    # classes = np.hstack(classes)

    return confidences, pred_is_tp


def pr_curve(pred_is_tp, confidences, total_positives):
    """
    Calculates precision and recall values for PR-curve.

    Parameters
    ----------
    pred_is_tp: ndarray of bool
        true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.
    confidences: ndarray of float
        predictions confidences
    total_positives: int
        total number of true positives

    Returns
    -------
    precisions: ndarray of float
        precision values
    recalls: ndarray of float
        recall values

    Note
    ----
    The PR-curve is defeined as follows.

    First all targets are sorted by their confidences in the descending order.
    Than the precision and recall values are calculated only for the nost confident target,
    than for two most confident targtes and so on.

    """
    pred_is_tp = np.copy(pred_is_tp)
    confidences = np.copy(confidences)

    ids_sorted = np.argsort(confidences)
    confidences = confidences[ids_sorted]
    pred_is_tp = pred_is_tp[ids_sorted]

    if not total_positives:
        raise Exception("No grounds truth labels present")

    total_preds = len(confidences)
    precisions = np.zeros((total_preds))
    recalls = np.zeros((total_preds))
    current_tps = 0

    for idx, is_tp in enumerate(pred_is_tp):
        if is_tp:
            current_tps += 1
        recalls[idx] = current_tps / total_positives
        precisions[idx] = current_tps / (idx + 1)

    return precisions, recalls


def interpolate_PR_curve(precisions, recalls):
    """
    Eliminates "wiggles" on the PR curve.

    Parameters
    ----------
    precisions: ndarray of float
        precision values
    recalls: ndarray of float
        recall values

    Returns
    -------
    precisions_interpolated:
        interpolated precision values with correspondence to raw precisions
    break_points:
        breaking points of the interpolated plot
    """
    last_precision = precisions[-1]
    precisions_interpolated = np.zeros_like(precisions)

    precisions_range = len(precisions)
    ids_to_change = []
    break_points = []

    for idx, precision in enumerate(precisions[::-1]):
        true_idx = precisions_range - idx - 1

        if precision < last_precision:
            ids_to_change.append(true_idx)
        else:
            precisions_interpolated[ids_to_change] = last_precision
            break_points.append(true_idx)

            ids_to_change = [true_idx]
            last_precision = precision

        if (
            len(break_points) > 1
            and recalls[break_points[-1]] == recalls[break_points[-2]]
        ):
            break_points.pop(-2)
        if (
            len(break_points) > 1
            and precisions[break_points[-1]] == precisions[break_points[-2]]
        ):
            break_points.pop(-1)

    if break_points[-1] == 0:
        break_points.pop()

    precisions_interpolated[ids_to_change] = last_precision
    return precisions_interpolated, break_points


def avg_precisions_from_composed(
    confidences, pred_is_tp, gt_classes, pred_classes, class_labels
):
    """
    Calculates average precisions (AP) for different classes from composed confidences and labels.
    
    Parameters
    ----------
    confidences: ndarray of float
        concatenated confidences of the targets
    pred_is_tp: ndarray of bool
        concatenated true positive flags for predictions. If prediction is not true positive, it is considered to be false positive.
    gt_classes: ndarray of float
        classes for ground truth targets
    pred_classes: ndarray of float
        classes for predicted targets
    class_labels: iterable of float
        class labels to consider 
    """
    avg_precs = {}
    curves = {}

    for class_i in class_labels:
        ids = np.where(pred_classes == class_i)[0]
        total_positives_gt = np.sum(gt_classes == class_i)
        total_positive_pred = np.sum(pred_is_tp[ids])
        if total_positives_gt == 0:
            warnings.warn("No true positives for class %i" % class_i)
            avg_precs[class_i] = None
        elif total_positive_pred == 0:
            avg_precs[class_i] = 0.0
            curves[class_i] = {
                "recalls": [0.0],
                "precisions": [0.0],
                "interpolated": [0.0],
            }
        else:
            precisions, recalls = pr_curve(
                pred_is_tp[ids], confidences[ids], total_positives_gt
            )
            precisions_interpolated, break_points = interpolate_PR_curve(
                precisions, recalls
            )
            avg_precs[class_i] = np.sum(
                precisions_interpolated[break_points]
                * (recalls[break_points] - recalls[break_points[1:] + [0]])
            )

            curves[class_i] = {
                "recalls": recalls,
                "precisions": precisions,
                "interpolated": precisions_interpolated,
            }

    return avg_precs, curves


def calculate_avg_precisions(
    similarity,
    sim_thresh,
    images_ids,
    targets_batched_gt,
    targets_batched_pred,
    confidences,
    class_labels,
):
    """
    Calculates average precisions (AP) for different classes.

    Parameters
    ----------
    similarity: endoanalysis.similarity.SimilarityMeasure
        a measure of similarity between targets
    sim_thresh: list float
        threshold for similarity between ground truth and prediction targets.
        If two targets have similarity below this threshold, they will not be mapped even if there are no better candidates
    images_ids: list of int
        ids of images to consider
    targets_batched_gt: endoanalysis.targets.ImageTargetsBatch
        ground truth targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    targets_batched_pred: endoanalysis.targets.ImageTargetsBatch
        prediction targets (like keypoints or bboxes) for a batch of images. Must be compatible with similarity.
    confidences: ndarray of float
        confdences of the predcitions
    class_labels: iterable of float
        class labels to consider

    Returns
    -------
    avg_precs: dict of float
        average precisions
    curves: dict of dict of ndarray
        PR curves for different classes.

    """
    confidences, pred_is_tp = compose_tp_labels(
        similarity,
        sim_thresh,
        images_ids,
        targets_batched_gt,
        targets_batched_pred,
        confidences,
    )

    classes_gt = np.hstack([targets_batched_gt.from_image(x).classes() for x in images_ids])
    classes_pred = np.hstack([targets_batched_pred.from_image(x).classes() for x in images_ids])

    avg_precs, curves = avg_precisions_from_composed(
        confidences,
        pred_is_tp,
        classes_gt,
        classes_pred,
        class_labels,
    )
    

    return avg_precs, curves