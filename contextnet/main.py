import torch
import os
import shutil
from contextnet.models.models import ContextNet
from contextnet.train.train import ModelTrainer
from contextnet.utils.datasets import PrecomputedDataset, HeatmapsDataset
from contextnet.utils.utils import parse_master_yaml
from nucleidet.data.datasets import PrecomputionLight
# from torch.utils.tensorboard import SummaryWriter


def main():
    # config_path = "configs/basic.yml" 
    # train_data_yaml = "../train/train.yaml"


    # # with open(config_path, "r") as config_file:
    # #     CONFIG = yaml.safe_load(config_file)

    # #augs_list = albumentations_from_config(CONFIG["augmentations"])
    # train_lists = parse_master_yaml(train_data_yaml)

    # train_dataset = HeatmapsDataset(
    #     train_lists["images_lists"],
    #     train_lists["heatmaps_lists"],
    #     train_lists["labels_lists"],
    #     num_classes=2,
    #     resize_to=(600, 600),
    #     heatmaps_shape=(152,152)
    # )
    # # for sample in train_dataset:
    # #     print()
    #     #break
    # model = PrecomputionLight(train_dataset, '../train_dataset')
    # model.make()
    # print()





    dataset = PrecomputedDataset('../train_dataset/data.txt')
    model = ContextNet()
    trainer = ModelTrainer(model, dataset)
    trainer.train()
    print()
    
    
    


    
    
if __name__ == '__main__':
    main()