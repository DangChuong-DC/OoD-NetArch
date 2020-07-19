import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from utils import *
from models.network import Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--eval_time', type=int, default=1, help='repetition of running evaluation')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='./CheckPoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=12345, help='random seed')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/Storage/ANAS/data/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

args, unparsed = parser.parse_known_args()

args.save = '{}supernet-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.cifar100:
    CIFAR_CLASSES = 100
    data_folder = 'cifar-100-python'
else:
    CIFAR_CLASSES = 10
    data_folder = 'cifar-10-batches-py'


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)

    # prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=False, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.tmp_data_dir, train=False, download=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers
        )

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers
        )

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    supernet = Network(args.init_channels, CIFAR_CLASSES, args.layers)
    supernet.cuda()

    optimizer = torch.optim.SGD(
        supernet.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])

        train_acc, train_obj = train(train_queue, supernet, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        valid_top1 = utils.AverageMeter()
        for i in range(args.eval_time):
            ops_alps = utils.get_child_alphas(supernet._layers, supernet._steps, free=False)
            subnet = supernet.get_sub_net(ops_alps)

            valid_acc, valid_obj = infer(valid_queue, subnet, criterion)
            valid_top1.update(valid_acc)
        logging.info('Mean Valid Acc: %f', valid_top1.avg)
        scheduler.step()

        utils.save(supernet, os.path.join(args.save, 'supernet_weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()
        model.generate_share_alphas()   # change to `generate_free_alphas()` to let cell's architecture free

        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)

    logging.info('Valid Stats --- Objs: %e Acc: %f', objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Running time: %ds.', duration)
