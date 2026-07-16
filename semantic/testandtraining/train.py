#!/usr/bin/env python3
"""Train the forestry RandLA-Net on data prepared by prepare_dataset.py.

Produces checkpoint.tar files compatible with the deployed inference engine
(include/inference.py) — deploy by copying the best checkpoint over
semantic/model/checkpoint.tar.

Example:
    python3 train.py --data ../data/prepared --out ../model/run1
    python3 train.py --data ../data/prepared --out /tmp/smoke --quick   # smoke test
"""
import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, join(os.path.dirname(BASE_DIR), 'include'))

from config import Config as cfg, LABEL_TO_NAME
from dataset import PreparedClouds, PatchSampler, to_device
from RandLANet import Network


def class_weights(counts):
    freq = counts / counts.sum()
    return 1.0 / (freq + 0.02)


def iou_from_confusion(conf):
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    return tp / np.maximum(denom, 1)


def run_epoch(net, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    net.train() if training else net.eval()
    conf = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    losses = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = to_device(batch, device)
            logits = net(batch)['logits']            # (B, C, N)
            labels = batch['labels']                 # (B, N)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pred = logits.argmax(dim=1).cpu().numpy().ravel()
            gt = labels.cpu().numpy().ravel()
            np.add.at(conf, (gt, pred), 1)
    return float(np.mean(losses)), conf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='prepared dataset root')
    ap.add_argument('--out', required=True, help='directory for checkpoints/logs')
    ap.add_argument('--epochs', type=int, default=cfg.max_epoch)
    ap.add_argument('--gpu', type=int, default=0, help='-1 for CPU')
    ap.add_argument('--resume', default='', help='checkpoint.tar to resume from')
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: 2 epochs of a few small steps')
    args = ap.parse_args()

    if args.quick:
        cfg.train_steps, cfg.val_steps, args.epochs = 4, 2, 2

    device = torch.device('cuda:%d' % args.gpu
                          if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    log = open(join(args.out, 'train_log.txt'), 'a')

    def say(msg):
        print(msg)
        log.write(msg + '\n')
        log.flush()

    train_clouds = PreparedClouds(args.data, 'training')
    val_clouds = PreparedClouds(args.data, 'validation')
    say('clouds: %d train, %d val' % (len(train_clouds.names), len(val_clouds.names)))

    counts = train_clouds.class_counts()
    weights = class_weights(counts)
    say('class counts: %s' % dict(zip([LABEL_TO_NAME[i] for i in range(4)], counts)))
    cfg.class_weights = np.expand_dims(weights, 0)

    train_loader = DataLoader(PatchSampler(train_clouds, 'training'),
                              batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=PatchSampler.collate_fn, num_workers=4)
    val_loader = DataLoader(PatchSampler(val_clouds, 'validation'),
                            batch_size=cfg.val_batch_size, shuffle=False,
                            collate_fn=PatchSampler.collate_fn, num_workers=2)

    net = Network(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).float().to(device))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        say('resumed from %s (epoch %d)' % (args.resume, start_epoch))

    best_miou = 0.0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, _ = run_epoch(net, train_loader, device, criterion, optimizer)
        val_loss, conf = run_epoch(net, val_loader, device, criterion)
        scheduler.step()

        iou = iou_from_confusion(conf)
        miou = float(np.mean(iou))
        oa = float(np.diag(conf).sum() / max(conf.sum(), 1))
        say('epoch %3d  train_loss %.4f  val_loss %.4f  OA %.3f  mIoU %.3f  [%s]  %.0fs'
            % (epoch, train_loss, val_loss, oa, miou,
               ' '.join('%s %.2f' % (LABEL_TO_NAME[i][:2], iou[i]) for i in range(4)),
               time.time() - t0))

        state = {'epoch': epoch,
                 'model_state_dict': net.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, join(args.out, 'checkpoint.tar'))
        if miou > best_miou:
            best_miou = miou
            torch.save(state, join(args.out, 'checkpoint_best.tar'))

    say('done. best val mIoU %.3f' % best_miou)
    say('deploy with: cp %s ../model/checkpoint.tar'
        % join(args.out, 'checkpoint_best.tar'))


if __name__ == '__main__':
    main()
