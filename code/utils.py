#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : utils.py
from __future__ import print_function

import argparse
import math
import os

import torch


def get_parser(stage="train"):
    """Parsing arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tensorboad-logdir", default="tblog")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--ckpt-dir", default="checkpoint")
    # encoder
    parser.add_argument("--modification", default="late", choices=['early', 'late'])
    parser.add_argument("--encoder-attention-heads", default=4, type=int)
    parser.add_argument("--encoder-embed-dim", default=256, type=int)
    parser.add_argument("--encoder-ffn-embed-dim", default=512, type=int)
    parser.add_argument("--encoder-layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    # decoder
    parser.add_argument("--node-embed-size", default=256, type=int)
    parser.add_argument("--node-hidden-size", default=256, type=int)
    parser.add_argument("--dec-layers", default=1, type=int)
    parser.add_argument("--edge-embed-size", default=256, type=int)
    parser.add_argument("--edge-hidden-size", default=256, type=int)

    # training
    if stage == "train":
        parser.add_argument("--epochs", default=20, type=int)
        parser.add_argument("--batch-size", default=64, type=int)
        parser.add_argument("--eval-step", default=5000, type=int)
        parser.add_argument("--lr", default=1e-2, type=float)
        parser.add_argument("--warmup", default=4000, type=int)
        parser.add_argument("--clip-norm", default=25.0, type=float)
        parser.add_argument("--accumulation-steps", default=9, type=int)

    if stage == "test":
        parser.add_argument('--greedy-search', action='store_true', help='disable progress bar')
        parser.add_argument("--batch-size", default=64, type=int)
        parser.add_argument("--max-nodes", default=15, type=int)

    return parser


class NoamOpt():
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, params):
        self.optimizer = optimizer
        self.params = params
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        else:
            return math.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = None
        self.optimizer.zero_grad()

    def load_opt(self, ckpt):
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self._step = ckpt["total_step"]

    def get_opt(self):
        return {"optimizer": self.optimizer.state_dict(),
                "total_step": self._step}

    def get_step(self):
        return self._step
        

def get_std_opt(args, model):
    """Build a optimizer"""
    return NoamOpt(args.encoder_embed_dim, 2, args.warmup,
                   torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9),
                   model.parameters())


def save_model(args, model, optimizer, cur_epoch, best_val, type="last"):
    """Save model and optimizer"""
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    path = os.path.join(args.ckpt_dir, "{}_model".format(type))

    saved = {'epoch': cur_epoch, 'model': model.state_dict(), 'best_val': best_val}
    saved.update(optimizer.get_opt())

    torch.save(saved, path)


def load_model(args, model, inference=False, optimizer=None):
    """Load saved model and optimizer from the specified path"""
    if inference:
        path = os.path.join(args.ckpt_dir, "best_model")
    else:
        path = os.path.join(args.ckpt_dir, "last_model")
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt['model'])
        if optimizer is not None:
            optimizer.load_opt(ckpt)

        return {"best_val": ckpt["best_val"], "epoch": ckpt["epoch"]}

    return {}


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    """Move a minibatch of data fro CPU to GPU"""
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
