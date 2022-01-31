import torch
from  contextnet.utils.losses import GaussianFocalLoss
from tqdm import tqdm
#from torch.utils.data import DataLoader
from torchvision.utils import save_image

class ModelTrainer:
    """

    """
    def __init__(self,
                 model,
                 dataset_train,
                 dataset_valid=None,
                 dataset_test=None,
                 optimizer=None,
                 loss=None,
                 scheduler=None,
                 epochs=300,
                 batch_size=5,
                 device=torch.device('cuda'),
                 val_step=1,
                 save_step=5,
                 test_after_train=False,
                 verbose=True):
        
        self.model = model
        self.model.to(device)
        # Dataloaders
        self.dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=dataset_train.collate_fn, multiprocessing_context='spawn')
        
        if dataset_valid is not None:
            self.dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)
            self.dataloader_valid = None
        else:
            self.valid_flg = 0
            
        if dataset_test is not None:
            self.dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
            self.dataloader_valid = None
        else:
            self.test_flg = 0
        
        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(params,
                                             lr=0.01)
        # Loss
        if loss is not None:
            self.loss_func = loss
        else:
            self.loss_func = GaussianFocalLoss
        
        # # Scheduler
        # if scheduler is not None:
        #     self.scheduler = scheduler
        # else:
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                      step_size=10,
        #                                                      gamma=0.1)
        self.epochs = epochs
        if val_step > 0 and type(val_step) == int:
            self.val_step = val_step
        else:
            self.val_step = None
        
        if save_step > 0 and type(save_step) == int:
            self.save_step = save_step
        else:
            self.save_step = None
        
        self.test_after_train = test_after_train
        self.verbose = verbose
        
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self._train_one_epoch(epoch)

            if self.val_step:
                if epoch % self.val_step == 0:
                    self._evaluate('val')
        
        if self.test_after_train:
            self._evaluate('test')
        
    def _evaluate(self, mode='val'):
        print()
        
    def test(self):
        self._evaluate('test')
        print()
    
    def _train_one_epoch(self, epoch):
        #self.model.train()
        loss_sum = 0
        for data in self.dataloader_train:
            self.optimizer.zero_grad()
            images = data['image']
            gt_keypoints = data['keypoints']
            gt_heatmaps = data['heatmap']

            pd_heatmaps = self.model(images)

            pd_image = (torch.cat((pd_heatmaps[0], torch.zeros(1, 152, 152).to('cuda')), 0)) / pd_heatmaps[0].max()
            gt_image = (torch.cat((gt_heatmaps[0], torch.zeros(1, 152, 152).to('cuda')), 0)) / gt_heatmaps[0].max()
            save_image(pd_image, 'pd.png')
            save_image(gt_image, 'gt.png')
            loss = self.model.get_loss(pd_heatmaps, gt_heatmaps)
            # print(loss)
            loss_sum += loss
            #self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            # self.scheduler.step()
        print(loss_sum)