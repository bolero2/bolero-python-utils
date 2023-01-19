# YOLOv5 YOLO-specific modules
import logging
import math
from copy import deepcopy
import os
import cv2
import time
from pathlib import Path
import numpy as np
from numpy import random
import yaml
from pprint import pprint

import torch
from torch.cuda import amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from models.common import *
from models.experimental import *

from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, labels_to_image_weights, check_file, check_img_size, \
    set_logging, one_cycle, colorstr, make_divisible
from utils.general import box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from utils.loss import ComputeLoss
from utils.torch_utils import ModelEMA
from utils.autoanchor import check_anchor_order
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.metrics import ap_per_class, ConfusionMatrix


try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

# sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

# device = torch.device('cpu')
# cuda = device.type != 'cpu'


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5m.yaml', ch=3, nc=None, anchors=None, job=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:
            with open(cfg) as f:
                self.yaml = yaml.load(cfg, Loader=yaml.SafeLoader)

        self.rank = self.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        self.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.plots = not self.yaml['train']['evolve']  # create plots

        self.job = job
        self.input_shape = self.job.target_size
        self.num_categories = self.job.num_categories
        self.category_names = self.job.category_names
        self.save_dir = self.job.save_dir

        self.iter_train_loss = []

        self.yaml['train']['project'] = self.save_dir

        print("YOLOv5: The number of classes: {}".format(self.num_categories))
        print("YOLOv5: Name of classes: {}".format(self.category_names))

        train_imgsz = self.input_shape[0]
        test_imgsz = self.input_shape[1]

        if train_imgsz not in [416, 640]:
            train_imgsz = 416
        
        if test_imgsz not in [416, 640]:
            test_imgsz = 416

        self.img_size = [train_imgsz, test_imgsz]

        # Define model
        ch = self.yaml['model']['ch'] = self.yaml['model'].get('ch', ch)  # input channels

        if nc and nc != self.yaml['model']['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['model']['nc']} with nc={nc}")
            # print(f"Overriding model.yaml nc={self.yaml['model']['nc']} with nc={nc}")
            self.yaml['model']['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            # print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['model']['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml['model']), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['model']['nc'])]  # default names

        # device, cuda setting
        self.is_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda") if self.is_cuda else torch.device('cpu')

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # self.best_fitness, self.training_results, self.optimizer, self.wandb_id, self.epoch = 0, 0, 0, 0, 0

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
        
    # @TODO
    def fit(self, x, y, validation_data, epochs=100, batch_size=1, callbacks=[]):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda' if is_cuda else 'cpu')
        self._device = _device

        self = self.to(_device)
        rank = self.global_rank
        save_dir = self.save_dir

        self.epochs = self.yaml['train']['epochs'] = epochs
        self.total_batch_size = self.batch_size = self.yaml['train']['batch_size'] = batch_size

        self.train_images = x
        self.train_annots = y
        self.valid_images = validation_data[0]
        self.valid_annots = validation_data[1]

        from callbacks import CustomGetMetricCallback, CustomSaveBestWeightCallback, CustomEarlyStopping

        my_getMetric = CustomGetMetricCallback(save_path=self.job.save_dir,
                                               _hash=self.job.hash)
        my_saveBestWeight = CustomSaveBestWeightCallback(save_path=self.job.save_dir,
                                                         _hash=self.job.hash,
                                                         lib='TORCH')
        my_earlySTOP = CustomEarlyStopping(patience=int(epochs * 0.1))

        total_train_iter = math.ceil(len(x) / self.batch_size)

        nc = 1 if self.yaml['train']['single_cls'] else self.num_categories  # number of classes
        names = self.category_names  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset.' % (len(names), nc)  # check

        # setting -> Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        self.yaml['hyp']['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        # print(f"Scaled weight_decay = {self.yaml['hyp']['weight_decay']}")
        logger.info(f"Scaled weight_decay = {self.yaml['hyp']['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        print('Learning Rate: %f' % float(self.yaml['hyp']['lr0']))
        if self.yaml['train']['adam']:
            print("Using Adam optimizer")
            optimizer = optim.Adam(pg0, lr=self.yaml['hyp']['lr0'], betas=(self.yaml['hyp']['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            print("Using SGD optimizer: [momentum={} | nesterov={}]".format(self.yaml['hyp']['momentum'], True))
            optimizer = optim.SGD(pg0, lr=self.yaml['hyp']['lr0'], momentum=self.yaml['hyp']['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.yaml['hyp']['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        if self.yaml['train']['linear_lr']:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - self.yaml['hyp']['lrf']) + self.yaml['hyp']['lrf']  # linear
        else:
            lf = one_cycle(1, self.yaml['hyp']['lrf'], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(self) if rank in [-1, 0] else None

        # Resume
        start_epoch, best_fitness = 0, 0.0

        # Image sizes
        gs = max(int(self.stride.max()), 32)  # grid size (max stride)
        nl = self.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in self.img_size]  # verify imgsz are gs-multiples

        # DP mode
        # if cuda and rank == -1 and torch.cuda.device_count() > 1:
        #     self = torch.nn.DataParallel(self)

        # SyncBatchNorm
        if self.yaml['train']['sync_bn'] and torch.cuda.is_available() and rank != -1:
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self).to(_device)
            logger.info('Using SyncBatchNorm()')
            # print("Using SyncBatchNorm()")

        trainpack = (x, y)
        testpack = validation_data

        """
        # (dc) [Default setting list]
        cache, rect, rank, world_size, workers, image_weights, quad

        train -> Augment = True
        valid(test) -> Augment = False
        """

        # Trainloader
        # (dc) check this part!!!!!!!!!!!

        dataloader, dataset = create_dataloader(trainpack,
                                                imgsz=imgsz, batch_size=batch_size, stride=gs,
                                                opt=self.yaml['train'], hyp=self.yaml['hyp'], augment=True, 
                                                cache=self.yaml['train']['cache_images'] and not self.yaml['train']['notest'], 
                                                rect=self.yaml['train']['rect'], rank=rank,
                                                world_size=self.world_size, workers=self.yaml['train']['workers'],
                                                image_weights=self.yaml['train']['image_weights'], 
                                                quad=self.yaml['train']['quad'], prefix=colorstr('train: '))

        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        assert mlc < nc, 'Label class %g exceeds nc=%g in datasets. Possible class labels are 0-%g' % (mlc, nc, nc - 1)

        # Process 0
        if rank in [-1, 0]:
            testloader = create_dataloader(testpack,  # testloader
                                           imgsz=imgsz_test, batch_size=batch_size * 2, stride=gs,
                                           opt=self.yaml['train'], hyp=self.yaml['hyp'], 
                                           cache=self.yaml['train']['cache_images'] and not self.yaml['train']['notest'], 
                                           rect=True, rank=-1,
                                           world_size=self.world_size, workers=self.yaml['train']['workers'],
                                           pad=0.5, prefix=colorstr('val: '))[0]

            if not self.yaml['train']['resume']:
                labels = np.concatenate(dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                # if plots:
                #     plot_labels(labels, names, save_dir, loggers)
                #     if tb_writer:
                #         tb_writer.add_histogram('classes', c, 0)

                # Anchors
                if not self.yaml['train']['noautoanchor']:
                    check_anchors(dataset, model=self, thr=self.yaml['hyp']['anchor_t'], imgsz=imgsz)
                self.half().float()  # pre-reduce anchor precision

        # DDP mode
        # if cuda and rank != -1:
        #     self = DDP(self, device_ids=[self.yaml['train']['local_rank']], output_device=self.yaml['train']['local_rank'],
        #                 # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
        #                 find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in self.modules()))

        # Model parameters
        self.yaml['hyp']['box'] *= 3. / nl  # scale to layers
        self.yaml['hyp']['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        self.yaml['hyp']['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.yaml['hyp']['label_smoothing'] = self.yaml['train']['label_smoothing']
        self.nc = nc  # attach number of classes to model
        self.hyp = self.yaml['hyp']  # attach hyperparameters to model
        self.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.class_weights = labels_to_class_weights(dataset.labels, nc).to(_device) * nc  # attach class weights
        self.names = names

        # Start training
        t0 = time.time()
        nw = max(round(self.yaml['hyp']['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=torch.cuda.is_available())
        compute_loss = ComputeLoss(self)  # init loss class
        logger.info(f'\n\n\nImage sizes              | {imgsz} train, {imgsz_test} test\n'
                    f'Using dataloader workers | {dataloader.num_workers} dataloader workers\n'
                    f'Logging results to       | {save_dir}\n'
                    f'Starting training for    | {epochs} epochs...\n\n')
        # print(f'Image sizes              | {imgsz} train, {imgsz_test} test\n'
        #       f'Using dataloader workers | {dataloader.num_workers} dataloader workers\n'
        #       f'Logging results to       | {save_dir}\n'
        #       f'Starting training for    | {epochs} epochs...')

        for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
            self.train()

            # Update image weights (optional)
            if self.yaml['train']['image_weights']:
                # Generate indices
                if rank in [-1, 0]:
                    cw = self.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

                # Broadcast if DDP
                if rank != -1:
                    indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        dataset.indices = indices.cpu().numpy()

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(4, device=_device)  # mean losses
            if rank != -1:
                dataloader.sampler.set_epoch(epoch)
            # pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
            # if rank in [-1, 0]:
            #     pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            start = time.time()
            for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(_device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / self.total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [self.yaml['hyp']['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.yaml['hyp']['warmup_momentum'], self.yaml['hyp']['momentum']])

                # Multi-scale
                if self.yaml['train']['multi_scale']:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=torch.cuda.is_available()):
                    pred = self(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(_device))  # loss scaled by batch_size
                    if rank != -1:
                        loss *= self.world_size  # gradient averaged between devices in DDP mode
                    if self.yaml['train']['quad']:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Osptimize
                if ni % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(self)

                # Print
                if rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    self.iter_train_loss = [i, mloss]
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    # s = ('%10s' * 2 + '%10.4g' * 6) % (
                    #     '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                    # pbar.set_description(s)

                    print("[train %s/%3s] Epoch: %3s | Time: %6.2fs | bbox_loss: %6.4f | object_loss: %6.4f | class_loss: %6.4f | total_loss: %6.4f" % (
                        i + 1, total_train_iter, epoch + 1, time.time() - start, *mloss))

                    # Plot
                    # if plots and ni < 3:
                    #     f = save_dir / f'train_batch{ni}.jpg'  # filename
                    #     Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    #     # if tb_writer:
                    #     #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                    # elif plots and ni == 10 and wandb_logger.wandb:
                    #     wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                    #                                 save_dir.glob('train*.jpg') if x.exists()]})

                # end batch ------------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------------
            print("[Epoch %3s training Ended] > Time: %6.2fs/epoch | bbox_loss: %6.4f | object_loss: %6.4f | class_loss: %6.4f | total_loss: %6.4f\n" % (
                        epoch + 1, time.time() - start, *mloss))

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
            scheduler.step()

            start = time.time()

            # DDP process 0 or single-GPU
            if rank in [-1, 0]:
                # mAP
                ema.update_attr(self, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                final_epoch = epoch + 1 == epochs
                if not self.yaml['train']['notest'] or final_epoch:  # Calculate mAP
                    # wandb_logger.current_epoch = epoch + 1
                    results, maps = evaluate(testpack,
                                             setting=self.yaml,
                                             batch_size=batch_size * 2,
                                             imgsz=imgsz_test,
                                             model=ema.ema,
                                             single_cls=self.yaml['train']['single_cls'],
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             verbose=nc < 50 and final_epoch,
                                             #  plots=plots and final_epoch,
                                             #  wandb_logger=wandb_logger,
                                             compute_loss=compute_loss,
                                             device=_device)
                                             #  is_coco=is_coco)
                    # result = (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps

                    mean_precision, mean_recall, mAP, mAP95, bbox_loss, obj_loss, cls_loss = results
                    loss = bbox_loss + obj_loss + cls_loss

                    print("[Epoch %3s validation Ended] > Time: %6.2fs/epoch | loss: %6.4f | mean Precision: %6.4f | mean Recall: %6.4f\n" % (
                        epoch + 1,  time.time() - start, loss, mean_precision, mean_recall))

                    # (dc) GetMetricCallback ================================================================================================
                    # bbox_loss = round(mloss.cpu().numpy().tolist()[0], 4) -> train metric
                    my_getMetric.save_metrics(epoch + 1, 
                                            metrics=[[round(mloss.cpu().numpy().tolist()[3], 4)],
                                                    [round(loss, 4), round(mAP, 4)]],
                                            metrics_name=[['loss'], ['loss', 'mAP']])
                    # # =======================================================================================================================

                    my_saveBestWeight.save_best_weight(model=self, model_metric=round(loss, 4), compared='less')

                    # # (dc) EarlyStoppingCallback ==============================================================================
                    my_earlySTOP(val_loss=round(loss, 4))

                    if my_earlySTOP.early_stop:
                        return "EARLY_STOP"
                    # =========================================================================================================

                # Write
                # with open(results_file, 'a') as f:
                #     f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
                # if len(opt.name) and opt.bucket:
                #     os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

                # Log
                # tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                #         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                #         'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                #         'x/lr0', 'x/lr1', 'x/lr2']  # params
                # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                #     if tb_writer:
                #         tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                #     if wandb_logger.wandb:
                #         wandb_logger.log({tag: x})  # W&B

                # Update best mAP
                # fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                # if fi > best_fitness:
                #     best_fitness = fi
                # wandb_logger.end_epoch(best_result=best_fitness == fi)

                # Save model
                # if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                #     ckpt = {'epoch': epoch,
                #             'best_fitness': best_fitness,
                #             'training_results': results_file.read_text(),
                #             'model': deepcopy(model.module if is_parallel(model) else model).half(),
                #             'ema': deepcopy(ema.ema).half(),
                #             'updates': ema.updates,
                #             'optimizer': optimizer.state_dict(),
                #             'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                #     # Save last, best and delete
                #     torch.save(ckpt, last)
                #     if best_fitness == fi:
                #         torch.save(ckpt, best)
                #     if wandb_logger.wandb:
                #         if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                #             wandb_logger.log_model(
                #                 last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                #     del ckpt

        # print("Fit part!")
        my_getMetric.visualize_losses()
        return results

    def predict(self, test_images, conf_thres=0.4, iou_thres=0.4, use_cpu=False):
        """
        # (dc) (important!!!!!!!!!) setting.yaml file update process
        - model save 방식이 model 전체를 저장하는 방식이여서
        training 당시에 model 객체에 저장된 멤버변수 값만 호출 가능함.

        - *model-id.py의 get_model을 사용하는 방식이 아님.*

        - 즉, weight를 불러와서 predict만 실행할 때, yaml 파일을 수정해도
        적용되지 않음. 
        
        # (참고사항)
        -> training 당시에 작성된 yaml 파일의 값으로 고정되는 것.
        -> 값을 수정하기 위해서는, predict 함수에 특정 값을 받아와서 주입하는 형식으로 들어가야 함.

        # (example) ===============================================================================
        opt = self.yaml['test']    # 여기서 self.yaml['test']의 값은, training당시에 설정된 값으로 들어감.
                                   # -> yaml을 수정한다고 해서 바뀌지 않음.

        # 수정 방법
        opt['classes'] = None
        opt['conf_thres'] = 0.77
        opt['iou_thres'] = 0.86     # 이런 방식으로, opt 지역 변수에 복사한 후 직접 값을 대입하는 방식으로 해야 함.
        # =========================================================================================
        """
        pred_result = []

        opt = self.yaml['test']
        
        opt['classes'] = None
        opt['conf_thres'] = conf_thres
        opt['iou_thres'] = iou_thres
        pprint(opt)

        # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.stride.max())  # model stride

        # source, weights, view_img, save_txt, imgsz = opt['source'], opt['weights'], opt['view_img'], opt['save_txt'], opt['img_size']
        imgsz = opt['img_size'] if len(set(self.img_size)) == 0 else self.img_size[-1]
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # save_img = not opt['nosave'] and not source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        # save_dir = Path(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['exist_ok']))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        # set_logging()
        # device = select_device(opt['device'])
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda and not use_cpu else torch.device('cpu')
        self = self.to(_device)
        self.eval()

        half = _device.type != 'cpu'  # half precision only supported on CUDA

        if half:
            self.half()  # to FP16

        # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        # vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

        dataset = LoadImages(test_images, img_size=imgsz, stride=stride)
        # Get names and colors
        names = self.module.names if hasattr(self, 'module') else self.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if _device.type != 'cpu':
            self(torch.zeros(1, 3, imgsz, imgsz).to(_device).type_as(next(self.parameters())))  # run once

        t0 = time.time()

        for path, img, im0s, shape, vid_cap in dataset:
            pred_output = []

            img = torch.from_numpy(img).to(_device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self(img, augment=opt['augment'])[0]

            # Apply NMS
            # opt['conf_thres'] = 0
            # opt['iou_thres'] = 0
            # print(opt['classes'])
            pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
            # t2 = time_synchronized()

            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # print("DET:", det)
                # print(len(det))
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # s += '(%g x %g) ' % img.shape[2:]  # print string
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det) != 0:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                    # Print results
                    # for c in set(det[:, -1].detach().cpu().numpy().tolist()):
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in reversed(det):
                        cls = int(cls.detach().cpu())
                        conf = float(conf.detach().cpu())
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        pred_output.append([cls, xywh[0], xywh[1], xywh[2], xywh[3], conf])
                        
                        # line = (cls, *xywh, conf) if opt['save_conf'] else (cls, *xywh)  # label format
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    #     if save_txt:  # Write to file
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if opt['save_conf'] else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #     if save_img or view_img:  # Add bbox to image
                    #         label = f'{names[int(cls)]} {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                else:
                    pred_output.append([0, 0, 0, 0, 0, 0])

                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                # if view_img:
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path != save_path:  # new video
                #             vid_path = save_path
                #             if isinstance(vid_writer, cv2.VideoWriter):
                #                 vid_writer.release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #                 save_path += '.mp4'
                #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer.write(im0)
            pred_result.append(pred_output)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {save_dir}{s}")

        print(f'Predict is Done. ({time.time() - t0:.3f}s)')
        # print(pred_result)
        # print("\n[PRED_RESULT]\n", pred_result)
        return pred_result

    # @TODO
    def load_weight(self, weights):
        checkpoint_file = weights
        print("YOLOv5: load weight from > {} ... Start".format(checkpoint_file))
        # self.load_state_dict(torch.load(checkpoint_file), strict=False)
        self = torch.load(checkpoint_file)
        self.eval()
        print("YOLOv5: load weight from > {} ... End".format(checkpoint_file))
        
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    # logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR, Shuffle_Block, conv_bn_relu_maxpool]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def evaluate(data,
             setting=None,
             weights=None,
             batch_size=32,
             imgsz=640,
             conf_thres=0.001,
             iou_thres=0.6,  # for NMS
             save_json=False,
             single_cls=False,
             augment=False,
             verbose=False,
             model=None,
             dataloader=None,
             save_dir=Path(''),  # for saving images
             save_txt=False,  # for auto-labelling
             save_hybrid=False,  # for hybrid auto-labelling
             save_conf=False,  # save auto-label confidences
             # plots=True,
             # wandb_logger=None,
             compute_loss=None,
             half_precision=True,
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
             # is_coco=False):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        _device = next(model.parameters()).device  # get model device

    # else:  # called directly
    #     set_logging()
    #     device = select_device(opt.device, batch_size=batch_size)

    #     # Directories
    #     save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #     # Load model
    #     model = attempt_load(weights, map_location=device)  # load FP32 model
    #     gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    #     imgsz = check_img_size(imgsz, s=gs)  # check img_size

    #     # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
    #     # if device.type != 'cpu' and torch.cuda.device_count() > 1:
    #     #     model = nn.DataParallel(model)

    # Half
    half = _device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    # half = False
    if half:
        model.half()

    # Configure
    model.eval()
    model = model.to(_device)
    # if isinstance(data, str):
    #     is_coco = data.endswith('coco.yaml')
    #     with open(data) as f:
    #         data = yaml.load(f, Loader=yaml.SafeLoader)
    # check_dataset(data)  # check
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    nc = 1 if single_cls else int(setting['model']['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(_device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    # log_imgs = 0
    # if wandb_logger and wandb_logger.wandb:
    #     log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    # if not training:
    #     if device.type != 'cpu':
    #         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    #     task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    #     dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
    #                                    prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # coco91class = coco80_to_coco91_class()
    # s = ('%7s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=_device)
    jdict, stats, ap, ap_class = [], [], [], []

    start = time.time()

    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(_device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(_device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            # t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            # t0 += time_synchronized() - t

            # Compute loss
            # if compute_loss:
            #     loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            now_loss = compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            loss += now_loss

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(_device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            # t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            # t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            pred = pred.to(_device)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            # if save_txt:
            #     gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
            #     for *xyxy, conf, cls in predn.tolist():
            #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #         with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
            #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            # if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
            #     if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
            #         box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
            #                      "class_id": int(cls),
            #                      "box_caption": "%s %.3f" % (names[cls], conf),
            #                      "scores": {"class_score": conf},
            #                      "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            #         boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            #         wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            # if save_json:
            #     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            #     image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            #     box = xyxy2xywh(predn[:, :4])  # xywh
            #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            #     for p, b in zip(pred.tolist(), box.tolist()):
            #         jdict.append({'image_id': image_id,
            #                       'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
            #                       'bbox': [round(x, 3) for x in b],
            #                       'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                # if plots:
                #     confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        print("[validation %s] | Time: %6.2fs | bbox_loss: %6.4f | object_loss: %6.4f | class_loss: %6.4f" % (
                batch_i + 1, time.time() - start, 
                now_loss.cpu().detach().numpy()[0],
                now_loss.cpu().detach().numpy()[1],
                now_loss.cpu().detach().numpy()[2]))

        # # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
        
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print("-> ", ('%7s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
    pf = '%7s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print("-> ", pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print("-> ", pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    # if not training:
    #     print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # # Plots
    # if plots:
    #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     if wandb_logger and wandb_logger.wandb:
    #         val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
    #         wandb_logger.log({"Validation": val_batches})
    # if wandb_images:
    #     wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # # Save JSON
    # if save_json and len(jdict):
    #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    #     anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
    #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    #     print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    #     with open(pred_json, 'w') as f:
    #         json.dump(jdict, f)

    #     try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval

    #         anno = COCO(anno_json)  # init annotations api
    #         pred = anno.loadRes(pred_json)  # init predictions api
    #         eval = COCOeval(anno, pred, 'bbox')
    #         if is_coco:
    #             eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
    #         eval.evaluate()
    #         eval.accumulate()
    #         eval.summarize()
    #         map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    #     except Exception as e:
    #         print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps# , t


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        images = [os.path.abspath(x) for x in path]
        ni = len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni  # number of files
        self.mode = 'image'
        self.cap = None
        assert self.nf > 0, f'No images found. '

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        shape = img0.shape[0:2]
        assert img0 is not None, 'Image Not Found ' + path
        # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, shape, self.cap

    def __len__(self):
        return self.nf  # number of files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='setting.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(_device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
