import torch
import numpy as np
import cv2
import os
import yaml
from skimage.morphology import convex_hull_image
import numpy as np
import json

import numpy as np


class SimilarityMeasure:
    """
    Base class for similarity measures
    """

    def measure(self, object1, object2):
        """
        Returns the measure value between two objects
        """
        return 0

    def matrix(self, container1, container2):
        """
        Returns the matrix of measure values between two sets of objects
        Sometimes can be implemented in a faster way than making couplewise measurements
        """
        matrix = np.zeros((len(container1), len(container2)))
        for i, object1 in enumerate(container1):
            for j, object2 in enumerate(container2):
                matrix[i, j] = self.measure(object1, object2)
        return matrix


class Minkovsky2DSimilarity(SimilarityMeasure):
    def __init__(self, p=2, scale=1.0):
        self.p = p
        self.scale = scale
        

    def measure(self, point1, point2):
        x_diff = point1.x_coords()[0] - point2.x_coords()[0]
        y_diff = point1.y_coords()[0] - point2.y_coords()[0]
        diffs = np.hstack([np.abs(x_diff), np.abs(y_diff)])
        powers = np.power(diffs, self.p)
        distance = np.power(powers.sum(), 1 / self.p) / self.scale

        return distance

    def matrix(self, points1, points2):

        coords1 = np.vstack([points1.x_coords(), points1.y_coords()])
        coords2 = np.vstack([points2.x_coords(), points2.y_coords()])

        diffs = np.abs(coords1[:, :, np.newaxis] - coords2[:, np.newaxis, :])

        powers = np.power(diffs, self.p)
        matrix = np.power(powers.sum(axis=0), 1 / self.p) / self.scale

        return matrix


class KPSimilarity(Minkovsky2DSimilarity):
    """
    Keypoints similarity
    """

    def __init__(self, p=2, scale=1.0, class_agnostic=True):
        super().__init__(p, scale)
        self.class_agnostic = class_agnostic

    def _exp_square(self, arr):
        return np.exp(-np.power(arr, 2) / 2.0)

    def measure(self, point1, point2):
        distance = super().measure(point1, point2)
        return self._exp_square(distance)

    def matrix(self, points1, points2):
        distance = super().matrix(points1, points2)
        matrix = self._exp_square(distance)
        if not self.class_agnostic:
            class_matrix = points1.classes().reshape(-1, 1) == points2.classes().reshape(1,-1)
            matrix = matrix * class_matrix
        return matrix

    

def prefilter_boxes(keypoints, skip_threshold=0.0):
    """
    Create dict with boxes stored by its label.

    Parameters
    ----------
    keypoints: np.array
        keypoints.
    skip_threshold: float 
        threshold for skipping keypoints with low confedences.
    
    Returns
    -------
    new_keypoints: dict
        dict with keypoints stored by its label.
    """
    new_keypoints = dict()

    for keypoint in keypoints:
        score = keypoint[3]
        if score < skip_threshold:
            continue
        label = int(keypoint[2])

        if label not in new_keypoints:
            new_keypoints[label] = []
        new_keypoints[label].append(keypoint)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_keypoints:
        current_boxes = sorted(
            new_keypoints[k], key=lambda x: x[3], reverse=True)
        new_keypoints[k] = np.stack(current_boxes)

    return new_keypoints


def get_weighted_box(points, conf_type='avg'):
    """
    Create weighted box for set of boxes.

    Parameters
    ----------
    points: np.array
        one keypoint.
    conf_type: str
        the type of confidence.

    Returns
    -------
    box: np.array
        new keypoint.
    """

    box = np.zeros(4, dtype=float)
    conf = 0
    conf_list = []
    for b in points:
        box[:2] += (b[3] * b[:2])
        conf += b[3]
        conf_list.append(b[3])
    box[2] = points[0][2]
    if conf_type == 'avg':
        box[3] = conf / len(points)
    elif conf_type == 'max':
        box[3] = np.array(conf_list).max()
    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[3] = conf / len(points)
    box[:2] /= conf
    return box


def find_matching_box(points, new_point, similarity, match_sim):
    """
    Compute similarity score between new points and all points to find most similar.

    Parameters
    ----------
    points: list
        The list of points for measering similarity.
    new_point: np.array
        The point which is wanted to match with one from points.
    similarity: func
        Similarity function.
    match_sim: float 
        threshold for similarity.
    Returns
    -------
    best_index: int
        index of most similar point in points to new_point.
    best_sim: float
        value of similarity between best similar point and new_point.
    """
    best_sim = match_sim
    best_index = -1
    for i in range(len(points)):
        box = points[i]
        sim = float(similarity.measure(
            box[:2], new_point[:2]))
        if sim > best_sim:
            best_index = i
            best_sim = sim
    return best_index, best_sim


def WBF(keypoints, similarity, threshold, skip_threshold=0.0, conf_type='avg'):
    '''
    Weighted Boxes Fusion method for matching keypoints.

    Parameters
    ----------
    keypoints: np.array
        input keypoints.
    similarity: func
        the function to measure similariry between keypoints.
    threshold: float
        threshold for similarity.
    skip_threshold: float
        the threshold to skip keypoints with low confidence.
    conf_type: str
        type of confidence.
    Returns
    -------
    match_keypoints: np.array
        matched keypoints.
    '''
    match_keypoints = []
    filtered_keypoints = prefilter_boxes(keypoints, skip_threshold)
    for label in filtered_keypoints:
        points = filtered_keypoints[label]
        new_points = []
        weighted_points = []
        for j in range(0, len(points)):
            index, best_sim = find_matching_box(
                weighted_points, points[j], similarity, threshold)
            if index != -1:
                new_points[index].append(points[j])
                weighted_points[index] = get_weighted_box(
                    new_points[index], conf_type)
            else:
                new_points.append([points[j].copy()])
                weighted_points.append(points[j].copy())
        match_keypoints += weighted_points
    match_keypoints = np.stack(match_keypoints)
    return match_keypoints


def extract_images_and_labels_paths(images_list_file, labels_list_file):

    images_list_dir = os.path.dirname(images_list_file)
    labels_list_dir = os.path.dirname(labels_list_file)

    with open(images_list_file, "r") as images_file:
        images = images_file.readlines()
        images = [
            os.path.normpath(os.path.join(images_list_dir, x.strip())) for x in images
        ]
    with open(labels_list_file, "r") as labels_file:
        labels = labels_file.readlines()
        labels = [
            os.path.normpath(os.path.join(labels_list_dir, x.strip())) for x in labels
        ]

    check_images_and_labels_pathes(images, labels)

    return images, labels


def agregate_images_and_labels_paths(images_lists, labels_lists):

    if type(images_lists) != type(labels_lists):
        raise Exception(
            "images_list_files and labels_list_file should have the same type"
        )

    if type(images_lists) != list:
        images_lists = [images_lists]
        labels_lists = [labels_lists]

    images_paths = []
    labels_paths = []
    for images_list_path, labels_list_path in zip(images_lists, labels_lists):
        images_paths_current, labels_paths_current = extract_images_and_labels_paths(
            images_list_path, labels_list_path
        )
        images_paths += images_paths_current
        labels_paths += labels_paths_current

    return images_paths, labels_paths


def check_images_and_labels_pathes(images_paths, labels_paths):

    if len(images_paths) != len(labels_paths):
        raise Exception("Numbers of images and labels are not equal")

    for image_path, labels_path in zip(images_paths, labels_paths):
        dirname_image = os.path.dirname(image_path)
        dirname_labels = os.path.dirname(labels_path)
        filename_image = os.path.basename(image_path)
        filename_labels = os.path.basename(labels_path)

        if ".".join(filename_image.split(".")[:-1]) != ".".join(
            filename_labels.split(".")[:-1]
        ):
            raise Exception(
                "Different dirnames found: \n %s\n  %s" % (images_paths, labels_paths)
            )


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_keypoints(file_path):
    """
    Load keypoints from a specific file as tuples

    Parameters
    ----------
    file_path : str
        path to the file with keypoints

    Returns
    -------
    keypoints : list of tuples
        list of keypoint tuples in format (x, y, obj_class)

    Note
    ----
    This function serves as helper for the pointdet.utils.dataset.PointsDataset class
    and probably should be moved there
    """

    keypoints = []

    with open(file_path, "r") as labels_file:
        for line in labels_file:
            line_contents = line.strip().split(" ")
            line_floated = tuple(int(float(x)) for x in line_contents)
            x_center, y_center, obj_class = tuple(line_floated)
            if obj_class == 2: # FIX ME
                obj_class = 0
            keypoint = x_center, y_center, obj_class
            keypoints.append(keypoint)

    return keypoints

def parse_master_yaml(yaml_path):
    """
    Imports master yaml and converts paths to make the usable from inside the script

    Parameters
    ----------
    yaml_path : str
        path to master yaml from the script

    Returns
    -------
    lists : dict of list of str
        dict with lists pf converted paths
    """
    with open(yaml_path, "r") as file:
        lists = yaml.safe_load(file)

    for list_type, paths_list in lists.items():
        new_paths_list = []
        for path in paths_list:
            new_path = os.path.join(os.path.dirname(yaml_path), path)
            new_path = os.path.normpath(new_path)
            new_paths_list.append(new_path)
        lists[list_type] = new_paths_list

    return lists

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

# def maker():
    #     images_pathes = []
    #     heatmaps_pathes = []
    #     a = set()
    #     n=0
    #     with open('../train/images_update.txt', 'r') as images_file:
    #         for path in images_file:
    #             n+=1
    #             path_norm = os.path.split(path.strip())[1]
    #             images_pathes.append(path.strip())
    #             heatmaps_pathes.append(os.path.join('./heatmaps', path_norm))
    #             shutil.copy(os.path.join('../train/', 'images_context', path_norm), os.path.join('../train/', './images', path_norm))
    #             shutil.copy(os.path.join('../train/', 'images_heatmaps', path_norm), os.path.join('../train/', './heatmaps', path_norm))
    #             a.add(path.strip())
    #     n = 0
    #     with open('../train/heatmaps.txt', 'w') as heatmaps_file:
    #         for path in heatmaps_pathes:
    #             heatmaps_file.write(f'{path}\n')
    #             n += 1
    #     n = 0
    #     with open('../train/images.txt', 'w') as images_file:
    #         for path in images_pathes:
    #             images_file.write(f'{path}\n')
    #             n += 1
    #     print()




#     images_pathes = []
#     labels_pathes = []
#     a = set()
#     labels = []
#     for (dirpath, dirnames, filenames) in walk('../../dataset_bulk/labels'):
#         labels.extend(filenames)
#         break

    
#     for name in labels:
#         name_norm = os.path.splitext(name)[0]
#         images_pathes.append(os.path.join('./images', f'{name_norm}.png'))
#         labels_pathes.append(os.path.join('./labels', f'{name_norm}.txt'))

#     with open('../../dataset_bulk/images.txt', 'w') as images_file:
#         for path in images_pathes:
#             images_file.write(f'{path}\n')
    
#     with open('../../dataset_bulk/labels.txt', 'w') as images_file:
#         for path in labels_pathes:
#             images_file.write(f'{path}\n')
#     print()


def compute_distances_no_loops(Y, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    dists = np.zeros((Y.shape[0], X.shape[0]))

    dists -= 2 * X @ Y.T
    dists += (np.sum(Y**2, axis=1))
    dists = dists.T + np.sum(X**2, axis=1)
    dists = dists.T
    return np.sqrt(dists)

def closest(x, dists):
    return np.argsort(dists[x, :])[1:3]


def generate_masks(image, keypoints):
    whole_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # whole_mask = np.zeros((170, 170),  dtype=np.uint8)
    # keypoints = keypoints.astype(int)
    if keypoints != []:
        # keypoints = np.array(epithelial, dtype=np.int32)
        dists = compute_distances_no_loops(keypoints, keypoints)
        for i, point in enumerate(keypoints):
            indices = closest(i, dists)
            if len(indices) > 0:
                j = indices[0]
            else:
                j = None
            if len(indices) > 1:
                k = indices[1]
            else:
                k = None
            tmp_mask = np.zeros(whole_mask.shape, dtype=np.uint8)
            tmp_mask = cv2.circle(tmp_mask, (keypoints[i][0], keypoints[i][1]), 16, 255, -1, lineType=cv2.LINE_AA)
            if j and dists[i, j] < 26:
                tmp_mask = cv2.circle(tmp_mask, (keypoints[j][0], keypoints[j][1]), 16, 255, -1, lineType=cv2.LINE_AA)
            if k and dists[i, k] < 36:
                tmp_mask = cv2.circle(tmp_mask, (keypoints[k][0], keypoints[k][1]), 16, 255, -1, lineType=cv2.LINE_AA)

            tmp_mask = convex_hull_image(tmp_mask)
            #helpers.image_show(255*tmp_mask)
            whole_mask = np.maximum(whole_mask, tmp_mask)
            #whole_mask = np.array(whole_mask, dtype=np.uint8)
    return whole_mask.astype(np.float32)
