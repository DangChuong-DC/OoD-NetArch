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
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--feat_comb', type=str, default='sum', help='type of feature combine method within cell')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=80, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='./results/', help='experiment path')
parser.add_argument('--load_at', type=str, default='./CheckPoints/supernet-cf100_8l_01lr_600e-20200720-135639/supernet_weights.pt', help='Checkpoint path.')
parser.add_argument('--super_seed', type=int, default=12345, help='random seed for supernet')
parser.add_argument('--folder', type=int, default=0, help='folder for saving')
parser.add_argument('--ckpt_path', type=str, default='/beegfs/anhcda/OoD_NAS/subnet_exp1/', help='path to save subnet weights')
parser.add_argument('--tmp_data_dir', type=str, default='/home/anhcda/Storage/OoD_NAS/data/', help='temp data dir')
parser.add_argument('--ood_dir', type=str, default='/home/anhcda/Storage/OoD_NAS/data/', help='Path of ood folder.')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--is_cosine', action='store_true', default=False, help='Specify if the cosine FC is used.')

args, unparsed = parser.parse_known_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, '{}/eval_out/{}/subnet_log.txt'.format(args.load_at.split('/')[2], args.folder)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

plot_pth = './results/{}/eval_out/{}/plot/tr/'.format(args.load_at.split('/')[2], args.folder)
writer_tr = SummaryWriter(plot_pth, flush_secs=30)
plot_pth = './results/{}/eval_out/{}/plot/va/'.format(args.load_at.split('/')[2], args.folder)
writer_va = SummaryWriter(plot_pth, flush_secs=30)
global_step = 0

if args.cifar100:
    CIFAR_CLASSES = 100
else:
    CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.super_seed)
    cudnn.benchmark = True
    torch.manual_seed(args.super_seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.super_seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)

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
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, drop_last=True)

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
    supernet = Network(
        args.init_channels, CIFAR_CLASSES, args.layers,
        combine_method=args.feat_comb, is_cosine=args.is_cosine,
    )
    supernet.cuda()
    supernet.generate_share_alphas()   #This is to prevent supernet alpha attribute being None type

    alphas_path = './results/{}/eval_out/{}/alphas.pt'.format(args.load_at.split('/')[2], args.folder)
    logging.info('Loading alphas at: %s' % alphas_path)
    alphas = torch.load(alphas_path)

    subnet = supernet.get_sub_net(alphas[:, :-1])
    logging.info(alphas)

    if args.cifar100:
        weight_decay = 5e-4
    else:
        weight_decay = 3e-4
    optimizer = torch.optim.SGD(
        subnet.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        logging.info('epoch {} lr {:.4f}'.format(epoch, scheduler.get_last_lr()[0]))

        train_acc, _ = train(train_queue, subnet, criterion, optimizer)
        logging.info('train_acc {:.2f}'.format(train_acc))

        valid_acc, valid_loss = infer(valid_queue, subnet, criterion)
        writer_va.add_scalar('loss', valid_loss, global_step)
        writer_va.add_scalar('acc', valid_acc, global_step)
        logging.info('valid_acc {:.2f}'.format(valid_acc))
        scheduler.step()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    utils.save(subnet, os.path.join(args.ckpt_path, 'subnet_{}_weights.pt'.format(args.folder)))

    lg_aucs, sm_aucs, ent_aucs = ood_eval(valid_queue, ood_queues, subnet, criterion)

    logging.info('Writting results:')
    out_dir = './results/{}/eval_out/{}/'.format(args.load_at.split('/')[2], args.folder)
    with open(os.path.join(out_dir, 'subnet_scratch.txt'), 'w') as f:
        f.write('-'.join([str(valid_acc), str(lg_aucs), str(sm_aucs), str(ent_aucs)]))


def get_mea(queue, net):
    net.eval()
    lgs, sms = [], []
    with torch.no_grad():
        for x, y in queue:
            if args.is_cosine:
                lg, cos = net(x.cuda(), get_cosine=True)
                lgs.append(cos.cpu().detach().numpy())
            else:
                lg = net(x.cuda())
                lgs.append(lg.cpu().detach().numpy())
            sm = torch.softmax(lg, 1)
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
        global global_step
        global_step += 1
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        model.zero_grad()

        logits = model(inp)
        loss = criterion(logits, target)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, _ = utils.accuracy(logits, target, topk=(1, 5))
        n = inp.size(0)
        objs.update(loss.clone().item(), n)
        top1.update(prec1.clone().item(), n)
        writer_tr.add_scalar('loss', loss.item(), global_step)
        writer_tr.add_scalar('acc', prec1.item(), global_step)
        decomp_loss_1 = torch.mean(torch.log(torch.sum(torch.exp(logits - torch.max(logits, 1)[0].view(-1, 1)), 1))).item()
        writer_tr.add_scalar('decomp_loss_lgsmex', decomp_loss_1, global_step)
        decomp_loss_2 = torch.mean(torch.max(logits, 1)[0] - logits[range(len(inp)), target.cpu().numpy()]).item()
        writer_tr.add_scalar('decomp_loss_max', decomp_loss_2, global_step)

        if (step + 1) % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %.2f', step + 1, objs.avg, top1.avg)

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

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Running time: {}s.'.format(duration))
