import torch
from  contextnet.utils.losses import GaussianFocalLoss, MixedLoss
from tqdm import tqdm
from nucleidet.detectors.heatmap import KeypointsExtractor
from torchvision.utils import save_image
from contextnet.eval.metrics import calculate_avg_precisions
from contextnet.utils.utils import KPSimilarity
from torchvision import transforms
from endoanalysis.visualization import visualize_keypoints
import matplotlib.pyplot as plt
from nucleidet.train.meters import mAPmetric
from torch.utils.tensorboard import SummaryWriter
from nucleidet.train.criterions import HeatmapHuber
from contextnet.utils.utils import _sigmoid



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
                 criterion=None,
                 scheduler=None,
                 similarity=None,
                 epochs=100,
                 batch_size=2,
                 device=None,
                 val_step=-1,
                 save_step=5,
                 save_path='./checkpoints',
                 test_after_train=False,
                 verbose=True,
                 classes=None,
                 kp_extractor=None):
        
        self.model = model
        # Dataloaders
        self.dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=dataset_train.collate_fn)
        
        if dataset_valid is not None:
            self.dataloader_val = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=dataset_valid.collate_fn)
        else:
            self.dataloader_val = None
            
        if dataset_test is not None:
            self.dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=dataset_test.collate_fn)
        else:
            self.dataloader_test = None
        
        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(params, lr=0.01)#, weight_decay=0.0001)
        # Loss
        if loss is not None:
            self.loss_func = loss
        else:
            # self.loss_func = HeatmapHuber(class_weights=(1.0, 1.0))
            # self.loss_func = MixedLoss(class_weights=(1.0, 1.0), focal_weight=0.002)
            self.loss_func = GaussianFocalLoss()
        

        # similarity
        if similarity is not None:
            self.similarity = similarity
        else:
            self.similarity = KPSimilarity(scale=7.62)

        # Scheduler
        self.scheduler = scheduler
        # if scheduler is not None:
        #     self.scheduler = scheduler
        # else:
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                      step_size=1000,
        #                                                      gamma=0.1)
        self.epochs = epochs
        if val_step > 0 and type(val_step) == int and self.dataloader_val is not None:
            self.val_step = val_step
        else:
            self.val_step = None
        
        if save_step > 0 and type(save_step) == int:
            self.save_step = save_step
        else:
            self.save_step = None
        
        self.save_path = save_path

        self.test_after_train = test_after_train
        self.verbose = verbose

        if classes is not None:
            self.classes = classes
        else:
            self.classes = [i for i in range(self.model.num_classes)]
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)

        self.kp_extractor = kp_extractor
        # if kp_extractor is not None:
        #     self.kp_extractor = kp_extractor
        # else:
        #     self.kp_extractor = self.model.get_keypoints

        # Criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = mAPmetric(self.similarity, range(len(self.classes)))

        # Logging init
        self.writer = SummaryWriter()
         

    def train(self):
        for epoch in range(self.epochs):
            # self.model.train()
            self._train_one_epoch(epoch)

            if self.val_step:
                if epoch % self.val_step == 0:
                    self.validate(epoch)
            if self.save_step:
                if epoch % self.save_step == 0:
                    torch.save(self.model, f"{self.save_path}/contextnet_{epoch}.pth")
        if self.test_after_train:
            self._evaluate('test')
    
    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_sum = 0
        n = 0
        for data in self.dataloader_train:
            
            images = data['image'].to(self.device)
            gt_keypoints = data['keypoints']
            gt_heatmaps = data['heatmap'].to(self.device)

            pd_heatmaps = self.model(images)
            # torch.onnx.export(self.model,               # model being run
            #       images,                         # model input (or a tuple for multiple inputs)
            #       "resnet_unet_modely.onnx")
            pd_heatmaps =_sigmoid(pd_heatmaps)
            pd_image = (torch.cat((pd_heatmaps[0], torch.zeros(1, 512, 512).to('cuda')), 0))
            gt_image = (torch.cat((gt_heatmaps[0], torch.zeros(1, 512, 512).to('cuda')), 0))
            # pd_image[0] = (pd_image[0] - pd_image[0].min()) / (pd_image[0].max() - pd_image[0].min())
            # pd_image[1] = (pd_image[1] - pd_image[1].min()) / (pd_image[1].max() - pd_image[1].min())
            save_image(images[0] / 255, f'./images_test/image_{n}.png')
            save_image(pd_image[0], f'./heatmaps_pd/pd_{n}_stroma.png')
            save_image(pd_image[1], f'./heatmaps_pd/pd_{n}_epith.png')
            save_image(gt_image, f'./heatmaps_gt/gt_{n}.png')
            
            loss = self.loss_func(pd_heatmaps.cpu(), gt_heatmaps.cpu())
            loss_sum += loss.item()
            n += 1
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.scheduler is not None:
            self.scheduler.step()
        loss_sum /= n
        self.writer.add_scalar('Loss/train', loss_sum, epoch)
        self.writer.add_scalar('lr', [group['lr'] for group in self.optimizer.param_groups][0], epoch)
        #print(loss_sum.item(), [group['lr'] for group in self.optimizer.param_groups])
        #self._logging_map(epoch + 1, loss_sum, [0.8131232, 0.6341341], [0.6451411, 0.431435135], [3456, 1633])
    
    def validate(self, epoch):
        loss_sum = self.evaluate(self.dataloader_val)
        # if self.verbose:
        #     self.eval_log(epoch, loss_sum, criterion_value)
        self.writer.add_scalar('Loss/val', loss_sum, epoch)
        # self.writer.add_scalar('Evaluation/val', criterion_value, epoch)

    def evaluate(self, dataloader):
        self.model.eval()
        #self.criterion.reset()
        loss_sum = 0
        n = 0
        with torch.no_grad():
            for data in dataloader:
                images = data['image']
                gt_keypoints = data['keypoints']
                gt_heatmaps = data['heatmap']
                pd_heatmaps = self.model(images.to(self.device))
                pd_heatmaps = pd_heatmaps.cpu()
                pd_heatmaps =_sigmoid(pd_heatmaps)
                # pd_image = (torch.cat((pd_heatmaps[0], torch.zeros(1, 256, 256)), 0))
                # gt_image = (torch.cat((gt_heatmaps[0], torch.zeros(1, 256, 256)), 0))
                # pd_image[0] = (pd_image[0] - pd_image[0].min()) / (pd_image[0].max() - pd_image[0].min())
                # pd_image[1] = (pd_image[1] - pd_image[1].min()) / (pd_image[1].max() - pd_image[1].min())
                # save_image(images[0] / 255, f'./images_test/image_{n}.png')
                # save_image(pd_image[0], f'./heatmaps_pd/pd_{n}_stroma.png')
                # save_image(pd_image[1], f'./heatmaps_pd/pd_{n}_epith.png')
                # save_image(gt_image, f'./heatmaps_gt/gt_{n}.png')
                loss = self.loss_func(pd_heatmaps, gt_heatmaps)
                loss_sum += loss.item()
                n += 1
                # pd_keypoints, confidences = self.kp_extractor(pd_heatmaps)
                # batch_gt = {"keypoints": gt_keypoints}
                # batch_pd = {"keypoints": pd_keypoints, "confidences": confidences}
                # self.criterion.update(batch_gt, batch_pd)
        loss_sum /= n
        #self.criterion.compute()
        #criterion_value = self.criterion.get_value()
        
        return loss_sum #, criterion_value
    
    def visualize(self, batch, ids=-1):
        # FIX ME
        images = batch['image']
        gt_keypoints = batch['keypoints']
        gt_heatmaps = batch['heatmap']
        len_batch = images.shape[0]
        image_transforms = transforms.Compose([
                                                transforms.CenterCrop(200),
                                                transforms.Resize(200)])
        
        self.model.eval()
        with torch.no_grad():
            pd_heatmaps = self.model(images)
            pd_keypoints, confidences = self.kp_extractor(pd_heatmaps)
            
            for k in range(len_batch):
                pd_heatmaps = pd_heatmaps.cpu()
                gt_heatmaps = gt_heatmaps.cpu()
                pd_image = ((torch.cat((pd_heatmaps[k], torch.zeros(1, 152, 152)), 0)) / pd_heatmaps[k].max()).permute(1, 2, 0)
                gt_image = ((torch.cat((gt_heatmaps[k], torch.zeros(1, 152, 152)), 0)) / gt_heatmaps[k].max()).permute(1, 2, 0)
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(gt_image)
                plt.show()
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(pd_image)
                plt.show()
                print('-----------------------------------------------------')
    
    def eval_log(self, epoch, loss, metrics):
        """
        +--------------------------------+
        | Epoch: 50/100   [///////     ] |
        | Loss:  630.7000 lr: 1.00E-03   |
        | mAP:   0.600                   |
        +--------------------------------+
        """
        sleshs = ''.join(["/" for i in range(int(epoch/self.epochs * 12))])
        spaces = ''.join([" " for i in range(12 - int(epoch/self.epochs * 12))])
        map_score = 0
        for ap in metrics:
            map_score += ap
        map_score /= len(metrics)
        print("+--------------------------------+")
        print("| Epoch: {0:>3d}/{1:<3d}  [{2}{3}] |".format(epoch, self.epochs, sleshs, spaces))
        print("| Loss:  {0:<8.3f} lr: {1:<.2E}{2}   |".format(loss,
                [group['lr'] for group in self.optimizer.param_groups][0]))
        print("| mAP:   {0:<5.3f}                   |".format(loss))
        print("+--------------------------------+")

    def _logging_map(self, epoch, loss, metrics, recall, cells_detected):
        """
        +-----------------------------------------+
        | Epoch: 50/100 [/////////////          ] |
        | Loss: 630.7       lr: 0.001             |
        +------------+----------------------------+
        |            |  AP      Recall  Detected  |
        | Stroma     |  0.600   0.300   0.900     |
        | Epithelium |  0.400   0.300   0.900     |
        +------------+----------------------------+
        | mAP        |  0.5                       |
        +------------+----------------------------+
        """
        sleshs = ''.join(["/" for i in range(int(epoch/self.epochs * self.bar_len))])
        spaces = ''.join([" " for i in range(self.bar_len - int(epoch/self.epochs * self.bar_len))])
        map_score = 0
        for ap in metrics:
            map_score += ap
        map_score /= len(metrics)
        print("+-----------------------------------------+")
        print("| Epoch: {0:>3d}/{1:<3d} [{2}{3}] |".format(epoch, self.epochs, sleshs, spaces))
        print("| Loss: {0:<9.3f}   lr: {1:<.2E}{2}|".format(loss, self.scheduler.get_last_lr(),
                ''.join(" " for i in range(self.table_len - 32)))) #FIX ME
        print("+{0}+{1}+".format("".join(["-" for i in range(self.class_max_len + 2)]), "".join(["-" for i in range(self.inner_log_len)])))
        print("|{0}|{1}|".format("".join([" " for i in range(self.class_max_len + 2)]), "  AP      Recall  Detected  "))
        for k, class_name in enumerate(self.classes):
            print("| {0}{1} |  {2:<6.3f}  {3:<6.3f}  {4:<.2E}  |".format(class_name,
                                                                        ''.join([" " for i in range(self.class_max_len - len(class_name))]),
                                                                        metrics[k],
                                                                        recall[k],
                                                                        cells_detected[k]))
        
        print("+{0}+{1}+".format("".join(["-" for i in range(self.class_max_len + 2)]), "".join(["-" for i in range(self.inner_log_len)])))
        print("| {0}{1} |  {2:<6.3f}{3}|".format('mAP',
                                            "".join(" " for i in range(self.class_max_len - len("mAP"))),
                                            map_score,
                                            "".join(" " for i in range(self.inner_log_len - 8))))
        print("+{0}+{1}+".format("".join(["-" for i in range(self.class_max_len + 2)]), "".join(["-" for i in range(self.inner_log_len)])))


