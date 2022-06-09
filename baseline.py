from __future__ import print_function

import argparse
import math
import os
import sys
import time
import pickle
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from losses import ContrastiveRanking
from networks.resnet_big import SupConResNet
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import str2bool
from demo.demo import run_test, get_parser, setup_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from torch.utils.data import Dataset, DataLoader
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

import xml.etree.ElementTree as ET

from testloader import TestDataLoader


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_fig', type=str2bool, default='False')
    parser.add_argument('--tsne', type=str2bool, default='False')
    parser.add_argument('--map', type=str2bool, default='False')
    parser.add_argument('--test', type=str2bool, default='True',
                        help='Test object detection')
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='./')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5, 
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'voc', 'path'], help='dataset')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # stuff for ranking
    parser.add_argument('--min_tau', default=0.1, type=float, help='min temperature parameter in SimCLR')
    parser.add_argument('--max_tau', default=0.2, type=float, help='max temperature parameter in SimCLR')
    parser.add_argument('--m', default=0.99, type=float, help='momentum update to use in contrastive learning')
    parser.add_argument('--do_sum_in_log', type=str2bool, default='True')
    parser.add_argument('--memorybank_size', default=2048, type=int)

    parser.add_argument('--similarity_threshold', default=0.01, type=float, help='')
    parser.add_argument('--n_sim_classes', default=1, type=int, help='')
    parser.add_argument('--use_dynamic_tau', type=str2bool, default='True', help='')
    parser.add_argument('--use_supercategories', type=str2bool, default='False', help='')
    parser.add_argument('--use_same_and_similar_class', type=str2bool, default='False', help='')
    parser.add_argument('--one_loss_per_rank', type=str2bool, default='True')
    parser.add_argument('--mixed_out_in', type=str2bool, default='False')
    parser.add_argument('--roberta_threshold', type=str, default=None,
                        help='one of 05_None; 05_04; 04_None; 06_None; roberta_superclass20; roberta_superclass_40')
    parser.add_argument('--roberta_float_threshold', type=float, nargs='+', default=None, help='')

    parser.add_argument('--exp_name', type=str, default=None, help='set experiment name manually')
    parser.add_argument('--mixed_out_in_log', type=str2bool, default='False', help='')
    parser.add_argument('--out_in_log', type=str2bool, default='False', help='')

    get_parser(parser)
    global opt
    opt = parser.parse_args()    

    if opt.dataset == 'cifar10':
        opt.num_classes = 10
    elif opt.dataset == 'cifar100':
        opt.num_classes = 100
    elif opt.dataset == 'voc':
        opt.size = 64
        opt.num_classes = 20
    else:
        raise ValueError("Dataset inappropriate")

    if opt.seed:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # the path according to the environment set
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_mit_{}_mat_{}_thr{}_cls_{}_memSize_{}'. \
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial, opt.min_tau, opt.max_tau,
               opt.similarity_threshold, opt.n_sim_classes, opt.memorybank_size)

    if opt.use_supercategories:
        opt.model_name = opt.model_name + '_superCat'
    if opt.use_same_and_similar_class:
        opt.model_name = opt.model_name + '_sim_class_sameRank'
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if not (opt.do_sum_in_log):
        opt.model_name = opt.model_name + 'log_out'
    if opt.mixed_out_in_log:
        opt.model_name = opt.model_name + 'mixed_log_out_in'

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.exp_name:
        opt.model_name = opt.exp_name

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    global img_size
    img_size = opt.size

    return opt

class VOC_collate:
    def __init__(self, train=False, mean=False):
        self.train = train
        self.mean = mean

    def __call__(self, batch):
        labels = []
        images_q = []
        images_k = []

        to_tensor = transforms.ToTensor()

        if self.train == True:
            train_transform = TwoCropTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.25),
                transforms.RandomGrayscale(p=0.2),
                transforms.Resize((img_size, img_size)),
                normalize
            ]))

        else:
            train_transform = TwoCropTransform(transforms.Compose([normalize,
                                                transforms.Resize((img_size, img_size))
                                                ]))

        original_images = []
        bboxes = []
        boxes_rcnn = []
        image_rcnn = []

        for i in batch:
            original_images.append(i[0])
            img_object = []
            i = list(i)
            image_rcnn.append(i[0])
            for obj in i[1]['annotation']['object']:
                if type(i[0]).__name__ == "Image":
                    i[0] = to_tensor(i[0])
                l = obj['name']
                x1 = int(obj['bndbox']['xmin'])
                x2 = int(obj['bndbox']['xmax'])
                y1 = int(obj['bndbox']['ymin'])
                y2 = int(obj['bndbox']['ymax'])
                img_object.append(torch.tensor([x1, y1, x2, y2]))
                img = i[0][:, y1:y2, x1:x2]
                img = train_transform(img)
                images_q.append(img[0])
                images_k.append(img[1])
                labels.append(l)
            img_object = torch.stack(img_object)
            bboxes.append(img_object)

        box_gt = torch.cat(bboxes, 0)

        images_q = torch.stack(images_q)
        images_k = torch.stack(images_k)

        images = [images_q, images_k]

        if self.train or self.mean or opt.tsne:
            return images, labels
        else:
            return original_images, images, labels, bboxes


def init_data_mean():
    if opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'cifar10':
        mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
        std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
    elif opt.dataset == 'voc':
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    global normalize
    normalize = transforms.Normalize(mean=mean, std=std)


def set_loader(opt):
    # construct data loader
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    val_transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
    ])

    if opt.dataset == 'voc':
        train_dataset = datasets.VOCDetection(root=opt.data_folder,
                            year='2007',
                            image_set='trainval',
                            download=False,
                            transform=None)
        mean_dataset = datasets.VOCDetection(root=opt.data_folder,
                            year='2007',
                            image_set='trainval',
                            download=False,
                            transform=None)
        val_dataset = datasets.VOCDetection(root=opt.data_folder,
                            year='2007',
                            image_set='test',
                            download=False,
                            transform=None)

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        mean_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=val_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                               train=False,
                               transform=val_transform)

    elif opt.dataset=='cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                     transform=TwoCropTransform(train_transform),
                                     download=False)
        mean_dataset = datasets.CIFAR10(root=opt.data_folder,
                                          transform=val_transform,
                                          download=False)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                               train=False,
                               transform=val_transform)

    else:
        raise ValueError(opt.dataset)


    ## {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    if opt.dataset != 'voc':
        opt.class_to_idx = train_dataset.class_to_idx
    
    elif opt.dataset == 'voc':
        opt.class_to_idx = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}

    print("Train Dataset size:", len(train_dataset))
    print("Mean Dataset size:", len(mean_dataset))
    print("Valid Dataset size:", len(val_dataset))
    # print("Test Dataset size:", len(test_dataset))

    if opt.dataset == 'voc':
        train_collate = VOC_collate(train=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True, collate_fn=train_collate)

        val_collate = VOC_collate(train=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, collate_fn=val_collate)

        mean_collate = VOC_collate(train=False, mean=True)
        mean_loader = torch.utils.data.DataLoader(
            mean_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, collate_fn=mean_collate)

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)

        mean_loader = torch.utils.data.DataLoader(
            mean_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True)

    ## {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    if opt.dataset != 'voc':
        opt.class_to_idx = train_dataset.class_to_idx
    
    elif opt.dataset == 'voc':
        opt.class_to_idx = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}


    if opt.test:
        return mean_loader, val_loader, mean_loader, opt
    else:
        return train_loader, opt


def train(train_loader, criterion, optimizer, epoch, opt):
    print(opt.num_classes)
    """one epoch training"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    loss_fn = torch.nn.CrossEntropyLoss()

    end = time.time()
    
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.dataset == 'voc':
            labels = torch.tensor([opt.class_to_idx[i] for i in labels])

        images = images[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        f1 = criterion(images)

        loss = loss_fn(f1, labels)

        # update metric
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def init_val_test(train_loader, val_loader, mean_loader, criterion, opt):
    mean_features = [[] for _ in range(opt.num_classes)]
    print(opt.num_classes)

    all_outputs = []
    all_labels = []

    for i, (images, labels) in enumerate(mean_loader):
        if opt.dataset == 'voc':
            images = images[0].cuda(non_blocking=True)
            for k in range(len(labels)):
                labels[k] = opt.class_to_idx[labels[k]]
        else:
            images = images.cuda(non_blocking=True)
        out = criterion(images).detach().cpu().numpy()
        all_outputs.extend(out)
        all_labels.extend(labels)

    label_count = [0 for _ in range(opt.num_classes)]

    for j in range(len(all_outputs)):
        if len(mean_features[all_labels[j]]) == 0:
            mean_features[all_labels[j]] = all_outputs[j]
        else:
            mean_features[all_labels[j]] += all_outputs[j]
        label_count[all_labels[j]] += 1

    mean_features = np.array(mean_features)
    mean_features = [mean_features[i]/label_count[i] for i in range(len(label_count))]
    mean_features = mean_features/np.sqrt(np.sum(np.square(mean_features), axis=1, keepdims=True))
    mean_features = torch.tensor(np.transpose(mean_features)).cuda()

    print("Mean Feature Representations computed")

    return mean_features


def load_saved_model(opt, criterion):
    ckpt = torch.load(opt.checkpoint)['model']
    criterion.load_state_dict(ckpt)


def test(opt):
    train_loader, val_loader, mean_loader, opt = set_loader(opt)
    criterion = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda()

    load_saved_model(opt, criterion)
    mean_features = init_val_test(train_loader, val_loader, mean_loader, criterion, opt)

    num_imgs_test = 0
    acc = 0

    if opt.dataset == 'voc':
        for idx, (original_images, images, labels, bboxes) in enumerate(val_loader):
            print(idx, len(val_loader))
            # box_rcnn, num_boxes_per_image, pred = run_test(opt, original_images)
            num_imgs_test += images[0].shape[0]
            images = torch.cat([images[0], images[1]], dim=0)
            labels = torch.tensor([opt.class_to_idx[i] for i in labels])

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            out = criterion(images[:len(labels)])
            out = out/torch.sqrt(torch.sum(torch.square(out), dim=1, keepdim=True))
            sim = torch.argmax(torch.matmul(out, mean_features), dim=1)

            img_pos = 0

            sim = torch.sum(torch.where(sim == labels, 1, 0))
            acc += sim.item()

        print("Test Accuracy: ", acc/num_imgs_test)


    elif opt.dataset == 'cifar10':
        for i, (images, labels) in enumerate(val_loader):
            print(i, len(val_loader))
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            num_imgs_valid += images.shape[0]
            out = criterion(images)
            out = out/torch.sqrt(torch.sum(torch.square(out), dim=1, keepdim=True))
            sim = torch.argmax(torch.matmul(out, mean_features), dim=1)
            sim = torch.sum(torch.where(sim == labels, 1, 0))
            acc += sim.item()

        print("Test Accuracy", acc/num_imgs_valid)



def plot_tsne():
    criterion = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda()
    val_dataset = datasets.VOCDetection(root=opt.data_folder,
                        year='2007',
                        image_set='test',
                        download=False,
                        transform=None)
    val_collate = VOC_collate(train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, collate_fn=val_collate)

    print("Plotting Tsne")
    all_outputs = []
    all_labels = []
    for i, (images, labels) in enumerate(val_loader):
        if opt.dataset == 'voc':
            images = images[0].cuda(non_blocking=True)
            for k in range(len(labels)):
                labels[k] = opt.class_to_idx[labels[k]]
        else:
            images = images.cuda(non_blocking=True)

        out = criterion(images).detach().cpu().numpy()
        all_outputs.extend(out)
        all_labels.extend(labels)


    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    all_outputs_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=25).fit_transform(all_outputs)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'lime', 'dodgerblue', 'chocolate']
    target_ids = range(opt.num_classes)
    for i, c, label in zip(target_ids, colors, all_labels):
        pos_tsne = np.where(all_labels==i)[0][:100]
        plt.scatter(all_outputs_embedded[pos_tsne, 0], all_outputs_embedded[pos_tsne, 1], c=c, label=label)
    plt.legend([i for i in opt.class_to_idx.keys()], ncol=4, fontsize='small')
    plt.savefig("./results/TSne_VOC2007_baseline.png")
    print(all_outputs_embedded.shape)



def main():
    opt = parse_option()

    opt = parse_option()

    init_data_mean()

    if opt.tsne:
        plot_tsne()
        return

    
    if opt.test == True:
        test(opt)
        return

    # build data loader
    train_loader, opt = set_loader(opt)

    # build model and criterion
    epoch = 1
    criterion = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).cuda()

    # build optimizer
    optimizer = torch.optim.SGD(criterion.parameters(),
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)

    start_epoch = 1
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cpu')
        criterion.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    # tensorboard
    tb_writer = SummaryWriter(log_dir=opt.tb_folder)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(criterion, optimizer, opt, epoch, save_file)


    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(criterion, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()




