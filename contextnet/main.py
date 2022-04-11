from scipy.fftpack import shift
import torch
import os
import shutil
from contextnet.models.models import ContextNet, ContextNetTry2, Segmentator
from contextnet.train.train import ModelTrainer, SegTrainer
from contextnet.utils.datasets import PrecomputedDataset, HeatmapsDataset, SegmentatedDataset
from contextnet.utils.utils import parse_master_yaml
from contextnet.utils.datasets import PrecomputionLight
# from torch.utils.tensorboard import SummaryWriter
from nucleidet.detectors.heatmap import KeypointsExtractor
import albumentations as A
from torchvision.utils import save_image
from nucleidet.train.meters import mAPmetric
from contextnet.utils.utils import KPSimilarity
from contextnet.utils.utils import _sigmoid
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU


def main():
    train_data_yaml = "/home/ushakov/isp/dataset_bulk_splitted/train.yaml"
    val_data_yaml = "/home/ushakov/isp/dataset_bulk_splitted/val.yaml"
    train_lists = parse_master_yaml(train_data_yaml)
    val_lists = parse_master_yaml(val_data_yaml)
    augs = [A.CoarseDropout(p=0.5),
        A.RandomGridShuffle(p=0.1),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.GridDistortion(p=0.5)]
    train_dataset = SegmentatedDataset(train_lists["images_lists"],
                    train_lists["labels_lists"],
                    n_classes=2,
                    augs_list=augs,
                    resize_to=(256,256))
    val_dataset = SegmentatedDataset(val_lists["images_lists"],
                    val_lists["labels_lists"],
                    n_classes=2,
                    resize_to=(256,256))
    
    model = Segmentator(activation="sigmoid")
    for param in model.model.encoder.parameters():
        param.requires_grad = False
        
    # model = torch.load('./checkpoints/contextnet_475.pth')
    loss = smp.losses.DiceLoss(mode='binary')
    criterion = IoU(activation='sigmoid')
    params =[p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0001)
    # optimizer = torch.optim.Adadelta(params, lr=0.05)
    trainer = SegTrainer(model,
        dataset_train=train_dataset,
        dataset_valid=val_dataset,
        loss=loss,
        criterion=criterion,
        batch_size=8,
        optimizer=optimizer,
        epochs=1500,
        val_step=5,
        verbose=False,
        classes=['Stroma', 'Epithelium'])
    trainer.train()
    # model.eval()
    # dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=val_dataset.collate_fn)
    # for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     criterion = IoU(threshold=threshold, activation='sigmoid')
    #     iou_sum = 0
    #     n = 0
    #     for data in dataloader:
    #         images = data['image'].to('cuda')
    #         gt_masks = data['mask'].to('cuda')
    #         pd_masks = model(images)
    #         # pd_masks = pd_masks.sigmoid()
    #         # pd_mask_mod = ((pd_masks[0] - pd_masks[0].min()) / (pd_masks[0].max() - pd_masks[0].min()))
    #         # pd_mask_max = (pd_mask_mod > 0.8) * 1.0
    #         # save_image(images[0] / 255, f'./images_test/image_{n}.png')
    #         # save_image(gt_masks[0], f'./masks_gt/mask_gt_{n}.png')
    #         # save_image(pd_masks[0].sigmoid(), f'./masks_pd/mask_pd_{n}.png')
            
    #         iou_sum += criterion(pd_masks, gt_masks)
    #         n += 1
    #     iou_sum /= n
    #     print(iou_sum, threshold)
    print()
    
if __name__ == '__main__':
    main()
