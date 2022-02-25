import os
import random
import torch
import shutil
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from matplotlib import cm
import albumentations as A
from endoanalysis.targets import Keypoints, keypoints_list_to_batch
from endoanalysis.visualization import visualize_keypoints
from nucleidet.data.heatmaps import make_heatmap
from nucleidet.data.keypoints import rescale_keypoints
from contextnet.utils.utils import agregate_images_and_labels_paths, load_image, load_keypoints
from endoanalysis.targets import KeypointsBatch
from torchvision.utils import save_image


class PrecomputedDataset(Dataset):
    """
    Dataset which loads the precomputed images, keypoints and heatmaps.
    """
    def __init__(self, data_file, epochs=1):
        self.data = []
        with open(data_file, 'r') as file:
            for data in file:
                self.data.append(data.strip())

        self.root_dir = os.path.split(data_file)[0]
        self.images_dir = 'images'
        self.keypoints_dir = 'keypoints'
        self.heatmaps_dir = 'heatmaps'
        if type(epochs) == int:
            self.epochs = [i for i in range(epochs)]
        else:
            self.epochs = epochs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        epoch = random.choice(self.epochs)
        image_name = os.path.join(self.root_dir, str(epoch), self.images_dir,
                                f'{self.data[idx]}.pt')
        keypoints_name = os.path.join(self.root_dir, str(epoch), self.keypoints_dir,
                                f'{self.data[idx]}.npy')
        heatmaps_name = os.path.join(self.root_dir, str(epoch), self.heatmaps_dir,
                                f'{self.data[idx]}.pt')
        
        image = torch.load(image_name)
        keypoints = np.load(keypoints_name)
        heatmaps = torch.load(heatmaps_name)
        sample = {'image': image, 'keypoints': keypoints, 'heatmap': heatmaps}

        return sample
    
    def collate_fn(self, data):
        images = []
        keypoints = []
        heatmaps = []
        for sample in data:
            images.append(sample['image'])
            keypoints.append(sample['keypoints'])
            heatmaps.append(sample['heatmap'])
        res = {'image': torch.stack(images), 'keypoints': KeypointsBatch(keypoints_list_to_batch(keypoints)), 'heatmap': torch.stack(heatmaps)}
        return res


class PointsDataset:
    def __init__(
        self,
        images_list,
        labels_list,
        keypoints_dtype=np.float,
        class_colors={x: cm.Set1(x) for x in range(10)},
    ):

        self.keypoints_dtype = keypoints_dtype

        self.images_paths, self.labels_paths = agregate_images_and_labels_paths(
            images_list,
            labels_list
        )
        self.class_colors = class_colors

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, x):

        image = load_image(self.images_paths[x])
        keypoints = load_keypoints(self.labels_paths[x])

        class_labels = [x[-1] for x in keypoints]
        keypoints_no_class = [x[:-1] for x in keypoints]

        keypoints = [
            np.array(y + (x,)) for x, y in zip(class_labels, keypoints_no_class)
        ]

        if keypoints:
            keypoints = np.stack(keypoints)
        else:
            keypoints = np.empty((0, 3))

        to_return = {"keypoints": Keypoints(keypoints.astype(self.keypoints_dtype))}

        to_return["image"] = image

        return to_return

    def visualize(
        self,
        x,
        show_labels=True,
        labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
    ):

        sample = self[x]

        image = sample["image"]

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            image,
            keypoints,
            class_colors=self.class_colors,
            circles_kwargs=labels_kwargs,
        )

    def collate(self, samples):

        images = [x["image"] for x in samples]
        keypoints_groups = [x["keypoints"] for x in samples]

        return_dict = {
            "image": np.stack(images, 0),
            "keypoints": keypoints_list_to_batch(keypoints_groups),
        }

        return return_dict


class HeatmapsDataset(PointsDataset, Dataset):
    "Dataset with images, keypoints and the heatmaps corresdonding to them."

    def __init__(
        self,
        images_list,
        labels_list,
        normalization=None,
        resize_to=None,
        keypoints_dtype=np.float,
        class_colors={x: cm.Set1(x) for x in range(10)},
        sigma=1,
        num_classes=1,
        augs_list=[],
        scale=1,
        heatmaps_shape=None,
        shift=(0, 0)
    ):

        super(HeatmapsDataset, self).__init__(
            images_list=images_list, labels_list=labels_list,
            class_colors=class_colors, keypoints_dtype=keypoints_dtype
        )

        self.sigma = sigma
        self.num_classes = num_classes
        self.heatmaps_shape = heatmaps_shape
        self.normalization = normalization
        self.scale = scale
        self.resize_to = resize_to
        if resize_to:
            augs_list.append(A.augmentations.Resize(*resize_to))

        self.alb_transforms = A.Compose(
            augs_list,
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["class_labels"]
            )
        )
        self.shift = shift

    def __getitem__(self, x):

        sample = super(HeatmapsDataset, self).__getitem__(x)
        y_size = sample["image"].shape[0] / self.scale
        x_size = sample["image"].shape[1] / self.scale
        
        if self.alb_transforms is not None:

            keypoints_no_class = np.stack(
                [sample["keypoints"].x_coords() + x_size,
                 sample["keypoints"].y_coords() + y_size]
            ).T
            classes = list(sample["keypoints"].classes())

            transformed = self.alb_transforms(
                image=sample["image"],
                keypoints=keypoints_no_class,
                class_labels=classes,
            )
            sample["image"] = transformed["image"]
            y_size = sample["image"].shape[0] / self.scale
            x_size = sample["image"].shape[1] / self.scale
            kp_coords = np.array(transformed["keypoints"])

            kp_coords[:, 0] = kp_coords[:, 0] - y_size
            kp_coords[:, 1] = kp_coords[:, 1] - x_size

            classes = np.array(transformed["class_labels"]).reshape(-1, 1)

            sample["keypoints"] = Keypoints(
                np.hstack([kp_coords, classes]).astype(float)
            )
            

        if self.heatmaps_shape:
            keypoints_to_heatmap = rescale_keypoints(
                sample["keypoints"], (y_size, x_size), self.heatmaps_shape
            )
            y_size, x_size = self.heatmaps_shape
        else:
            keypoints_to_heatmap = sample["keypoints"]

        sample["heatmaps"] = make_heatmap(
            x_size, y_size, keypoints_to_heatmap, self.sigma, self.num_classes
        )

        sample["image"] = np.moveaxis(sample["image"], -1, 0)

        

        for key in ["heatmaps", "image"]:
            sample[key] = torch.tensor(sample[key]).float()

        if self.normalization:
            sample["image"] -= torch.tensor(self.normalization["mean"]).reshape(
                -1, 1, 1
            )
            sample["image"] /= torch.tensor(self.normalization["std"]).reshape(-1, 1, 1)

        # gt_image = (torch.cat((sample["heatmaps"], torch.zeros(1, 512, 512)), 0))
        # save_image(sample["image"] / 255, f'./images_test/image_{x}.png')
        # save_image(gt_image, f'./heatmaps_gt/gt_{x}.png')

        return sample

    def collate(self, samples):

        images = [x["image"] for x in samples]
        keypoints_groups = [x["keypoints"] for x in samples]
        heatmaps = [x["heatmaps"] for x in samples]

        return_dict = {
            "image": torch.stack(images, 0).contiguous(),
            "keypoints": keypoints_list_to_batch(keypoints_groups),
            "heatmaps": torch.stack(heatmaps, 0).contiguous(),
        }

        return return_dict

    def visualize(
        self,
        x,
        show_labels=True,
        labels_kwargs={"radius": 3, "alpha": 1.0, "ec": (0, 0, 0)},
    ):

        sample = self[x]
        if self.normalization:
            sample["image"] = sample["image"] * torch.tensor(
                self.normalization["std"]
            ).view(-1, 1, 1) + torch.tensor(self.normalization["mean"]).view(-1, 1, 1)
        sample["image"] = sample["image"].int().numpy()
        sample["image"] = np.moveaxis(sample["image"], 0, -1)

        if show_labels:
            keypoints = sample["keypoints"]
        else:
            keypoints = Keypoints(np.empty((0, 3)))

        _ = visualize_keypoints(
            sample["image"],
            keypoints,
            class_colors=self.class_colors,
            circles_kwargs=labels_kwargs,
        )

class PrecomputionLight:
    """
    Precomputes dataset with augmentations.

    PrecomputionLight take the Dataset and output dir, in which dataset will be saved.
    PrecomputionLigh iters throught entire Dataset k times and saves
    all images, keypoints and heatmaps as files in outputdir.

    Parameters
    ----------
    dataset: Dataset
        the instance of the Dataset class.
    output_dir: str
        path to the output dir.
    k: int
        count of transforms apllications.
    overwrite: bool 
        the flag of overwriting files.
    """

    def __init__(self, dataset, output_dir, k=1, overwrite=False):
        self.dataset = dataset
        self.output_dir = output_dir
        self.k = k
        self.overwrite = overwrite
        
        self.indexes = set()
        self.images_dir = 'images'
        self.keypoints_dir = 'keypoints'
        self.heatmaps_dir = 'heatmaps'
        self.data_name = 'data'


    def make(self):
        """
        Iter throught dataset and save all features to the output dir.
        """
        self._make_dir(self.output_dir)
        
        for i in range(self.k):
            self._make_dir(os.path.join(self.output_dir, str(i)))
            self._make_dir(os.path.join(self.output_dir, str(i), self.images_dir))
            self._make_dir(os.path.join(self.output_dir, str(i), self.keypoints_dir))
            self._make_dir(os.path.join(self.output_dir, str(i), self.heatmaps_dir))

        for index in tqdm(range(len(self.dataset))):
            self.indexes.add(index)
            for i in range(self.k):
                data = self.dataset[index]
                image = data['image']
                keypoints = data['keypoints']
                heatmaps = data['heatmaps']
                
                torch.save(image, f'{os.path.join(self.output_dir, str(i), self.images_dir, str(index))}.pt')
                np.save(f'{os.path.join(self.output_dir, str(i), self.keypoints_dir, str(index))}.npy', keypoints)
                torch.save(heatmaps, f'{os.path.join(self.output_dir, str(i), self.heatmaps_dir, str(index))}.pt')

        with open(f'{os.path.join(self.output_dir, self.data_name)}.txt', 'w') as data_file:
            for index in self.indexes:
                data_file.write(f'{index}\n')
        

    def _make_dir(self, dir):
        """
        Making new directory.

        Parameters
        ----------
        dir: str 
            path to directory.
        """
        if os.path.isdir(dir):
            if self.overwrite:
                shutil.rmtree(dir)
            else:
                raise Exception('Output directory is not empty and overwrite flag is disabled, aborting.')
        os.makedirs(dir)