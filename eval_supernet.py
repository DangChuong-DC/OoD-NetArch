import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import sklearn.metrics as sk
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import warnings

from utils import *
from models.network import Network


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--feat_comb', type=str, default='sum', help='type of feature combine method within cell')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='./CheckPoints/', help='experiment path')
parser.add_argument('--load_at', type=str, default='./CheckPoints/supernet-try-20200719-183431/supernet_weights.pt', help='Checkpoint path.')
parser.add_argument('--seed', type=int, default=9, help='random seed')
parser.add_argument('--tmp_data_dir', type=str, default='/home/engkarat/data/storage/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--fine_tune', action='store_true', default=False, help='Specify if fine-tuning is done.')
parser.add_argument('--ood_dir', type=str, default='/home/engkarat/data/storage/ood_datasets_for_cifar/', help='Path of ood folder.')

args, unparsed = parser.parse_known_args()

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'


def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print("args = %s", args)
    print("unparsed args = %s", unparsed)

    # prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    ood_queues = {}
    for k in ['svhn', 'lsun_resized', 'imnet_resized']:
        ood_path = os.path.join(args.ood_dir, k)
        dset_ = dset.ImageFolder(ood_path, valid_transform)
        loader = torch.utils.data.DataLoader(
            dset_, batch_size=args.batch_size, shuffle=False,
            pin_memory=True, num_workers=args.workers
        )
        ood_queues[k] = loader

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    supernet = Network(args.init_channels, CIFAR_CLASSES, args.layers, combine_method=args.feat_comb)
    supernet.cuda()
    # print(len(supernet.cells))
    ckpt = torch.load(args.load_at)
    print(args.load_at)
    supernet.load_state_dict(ckpt)
    supernet.generate_share_alphas()
    # alphas = torch.Tensor([
    #     [0., 1., 1.],
    #     [0., 1., 0.],
    #     [0., 1., 0.],
    #     [0., 1., 1.],
    #     [0., 1., 1.],
    #     [0., 1., 1.],
    #     [0., 1., 0.],
    #     [0., 1., 0.],
    #     [0., 1., 1.],
    #     [0., 1., 0.],
    #     [0., 1., 0.],
    #     [0., 1., 1.],
    #     [0., 1., 0.],
    #     [0., 1., 1.]
    # ]).cuda()
    # for i in range(8):
    #     supernet.cells[i].ops_alphas = alphas
    alphas = supernet.cells[0].ops_alphas
    print(alphas)
    out_dir = './eval_out/{}'.format(args.seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(alphas, os.path.join(out_dir, 'alphas.pt'))
    with open(os.path.join(out_dir, 'alphas.txt'), 'w') as f:
        for i in alphas.cpu().detach().numpy():
            for j in i:
                f.write('{:d}'.format(int(j)))
            f.write('\n')

    if args.cifar100:
        weight_decay = 5e-4
    else:
        weight_decay = 3e-4
    optimizer = torch.optim.SGD(
        supernet.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=0)

    valid_acc, _ = infer(valid_queue, supernet, criterion)
    print('valid_acc {:.2f}'.format(valid_acc))
    lg_aucs, sm_aucs, ent_aucs = ood_eval(valid_queue, ood_queues, supernet, criterion)
    with open(os.path.join(out_dir, 'before.txt'), 'w') as f:
        f.write('-'.join([str(valid_acc), str(lg_aucs), str(sm_aucs), str(ent_aucs)]))

    if args.fine_tune:
        for epoch in range(args.epochs):
            # scheduler.step()
            print('epoch {} lr {:.4f}'.format(epoch, 0.001))#scheduler.get_lr()[0]))

            train_acc, _ = train(train_queue, supernet, criterion, optimizer)
            print('train_acc {:.2f}'.format(train_acc))

            valid_acc, _ = infer(valid_queue, supernet, criterion)
            print('valid_acc {:.2f}'.format(valid_acc))

        lg_aucs, sm_aucs, ent_aucs = ood_eval(valid_queue, ood_queues, supernet, criterion)
        with open(os.path.join(out_dir, 'after.txt'), 'w') as f:
            f.write('-'.join([str(valid_acc), str(lg_aucs), str(sm_aucs), str(ent_aucs)]))


def get_mea(queue, net):
    net.eval()
    lgs, sms = [], []
    with torch.no_grad():
        for x, y in queue:
            lg = net(x.cuda())
            sm = torch.softmax(lg, 1)
            lgs.append(lg.cpu().detach().numpy())
            sms.append(sm.cpu().detach().numpy())
    lgs = np.concatenate(lgs, 0)
    sms = np.concatenate(sms, 0)
    return lgs, sms


def auroc(mi, mo):
    score = np.concatenate([mi, mo])
    y_true = np.concatenate([np.ones(len(mi)), np.zeros(len(mo))])
    return sk.roc_auc_score(y_true, score) * 100


def get_ood_auroc(lgs_id, sms_id, lgs_od, sms_od):
    # max LG
    auc1 = auroc(np.max(lgs_id, 1), np.max(lgs_od, 1))
    # max SM
    auc2 = auroc(np.max(sms_id, 1), np.max(sms_od, 1))
    # Entropy
    neg_ent_id = np.sum(sms_id * np.log(sms_id), 1)
    neg_ent_id[np.isnan(neg_ent_id)] = 0
    neg_ent_od = np.sum(sms_od * np.log(sms_od), 1)
    neg_ent_od[np.isnan(neg_ent_od)] = 0
    auc3 = auroc(neg_ent_id, neg_ent_od)
    return auc1, auc2, auc3


def ood_eval(valid_queue, ood_queues, net, criterion):
    lg_aucs, sm_aucs, ent_aucs = [], [], []
    lgs_id, sms_id = get_mea(valid_queue, net)
    for k in ood_queues.keys():
        lgs_od, sms_od = get_mea(ood_queues[k], net)
        lg_auc, sm_auc, ent_auc = get_ood_auroc(lgs_id, sms_id, lgs_od, sms_od)
        lg_aucs.append(lg_auc); sm_aucs.append(sm_auc); ent_aucs.append(ent_auc)
    return np.mean(lg_aucs), np.mean(sm_aucs), np.mean(ent_aucs)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (inp, target) in enumerate(train_queue):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()

        logits = model(inp)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)
        step += 1

        if step % args.report_freq == 0:
            print('Train Step: {:3d} Objs: {:.4f} Acc: {:.2f}'.format(step, objs.avg, top1.avg))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    for step, (inp, target) in enumerate(valid_queue):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(inp)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)
        step += 1

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print('Running time: {}s.'.format(duration))
