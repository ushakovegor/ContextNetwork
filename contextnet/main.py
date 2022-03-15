from scipy.fftpack import shift
import torch
import os
import shutil
from contextnet.models.models import ContextNet, ContextNetTry2
from contextnet.train.train import ModelTrainer
from contextnet.utils.datasets import PrecomputedDataset, HeatmapsDataset
from contextnet.utils.utils import parse_master_yaml
from contextnet.utils.datasets import PrecomputionLight
# from torch.utils.tensorboard import SummaryWriter
from nucleidet.detectors.heatmap import KeypointsExtractor
import albumentations as A
from torchvision.utils import save_image
from nucleidet.train.meters import mAPmetric
from contextnet.utils.utils import KPSimilarity
from contextnet.utils.utils import _sigmoid


def main():
    # alb = [
    #     A.RandomBrightnessContrast(p=0.2),
    #     # A.RandomGridShuffle(p=0.2),
    #     A.Flip(),
    #     A.RandomRotate90()
    #     ]
    # train_data_yaml = "/home/ushakov/isp/dataset_bulk_splitted/train.yaml"
    # train_lists = parse_master_yaml(train_data_yaml)

    # train_dataset = HeatmapsDataset(
    #     train_lists["images_lists"],
    #     train_lists["labels_lists"],
    #     num_classes=2,
    #     resize_to=(512, 512),
    #     scale=3,
    #     heatmaps_shape=(512, 512),
    #     sigma=7.62,
    #     augs_list=alb
    # )
    # # for sample in train_dataset:
    # #     print()
    #     #break
    # model = PrecomputionLight(train_dataset, '../../dataset_bulk_train_precomputed', k=10, overwrite=True)
    # model.make()
    # print()

    train_dataset = PrecomputedDataset('../../dataset_bulk_train_precomputed/data.txt', epochs=10)
    val_dataset = PrecomputedDataset('../../dataset_bulk_val_precomputed/data.txt')
    # model = ContextNet()
    # # model = torch.load('./checkpoints/contextnet_220.pth')
    # # extractor = KeypointsExtractor(0.01, 9, (200, 200), 5)
    # trainer = ModelTrainer(model, dataset_train=train_dataset, dataset_valid=val_dataset, kp_extractor=None, epochs=3000, val_step=5, verbose=False, classes=['Stroma', 'Epithelium'])
    # trainer.train()
    # # loss_sum, criterion_value = trainer.evaluate(dataloader=trainer.dataloader_val)
    # # print(loss_sum, criterion_value)
    # print()
    
    model = torch.load('./models/contextnet_40 (copy)_focal_loss.pth')
    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=train_dataset.collate_fn)
    extractor = KeypointsExtractor(0.11, 9, (512, 512), 7)
    model.eval()
    similarity = KPSimilarity(scale=7.62)
    criterion = mAPmetric(similarity, (0, 1))
    criterion.reset()
    for data in dataloader:
        images = data['image']
        gt_keypoints = data['keypoints']
        gt_heatmaps = data['heatmap']
        pd_heatmaps = model(images.to('cuda:0'))
        pd_heatmaps = _sigmoid(pd_heatmaps)
        pd_image = (torch.cat((pd_heatmaps[0], torch.zeros(1, 512, 512).to('cuda')), 0))
        gt_image = (torch.cat((gt_heatmaps[0], torch.zeros(1, 512, 512)), 0))
        pd_image[0] = (pd_image[0] - pd_image[0].min()) / (pd_image[0].max() - pd_image[0].min())
        pd_image[1] = (pd_image[1] - pd_image[1].min()) / (pd_image[1].max() - pd_image[1].min())
        save_image(images[0] / 255, f'./images_test/image_test.png')
        save_image(pd_image[0], f'./heatmaps_pd/pd_stroma.png')
        save_image(pd_image[1], f'./heatmaps_pd/pd_epith.png')
        save_image(gt_image, f'./heatmaps_gt/gt_test.png')
        # print()
        pd_keypoints, confidences = extractor(pd_heatmaps)
        batch_gt = {"keypoints": gt_keypoints}
        batch_pd = {"keypoints": pd_keypoints, "confidences": confidences}
        criterion.update(batch_gt, batch_pd)
        
    criterion.compute()
    criterion_value = criterion.get_value()
    print(criterion_value)
    print()
        

    
    
if __name__ == '__main__':
    main()
