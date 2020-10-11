#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : train.py
from __future__ import print_function

import os
import time

import numpy as np
import torch

from torch.utils.data import DataLoader

from utils import get_parser, save_model, load_model, get_std_opt, move_to_cuda
from models import GraphTrans
from data_utils import Dictionary, GraphReader, TextReader, GraphTransReader, BatchSampler
from search import greedy_search


def load_dict(args):
    """ Load dictionary """
    node_dict = Dictionary().load(os.path.join(args.data_dir, "dict.txt"))

    node_dict.add_symbol("<blank>")
    edge_dict = node_dict
    text_dict = node_dict

    return node_dict, edge_dict, text_dict


def load_data(args, node_dict, edge_dict, text_dict, stage="train"):
    """ Load data
    stage: train/dev
    """
    src_graph = GraphReader(os.path.join(args.data_dir, "{}_src_graph.bin".format(stage)), node_dict, edge_dict)
    src_text = TextReader(os.path.join(args.data_dir, "{}_src_text.txt".format(stage)), text_dict)
    tgt_graph = GraphReader(os.path.join(args.data_dir, "{}_tgt_graph.bin".format(stage)), node_dict, edge_dict)
    data = GraphTransReader(src_graph, src_text, tgt_graph,
                            node_dict.pad(), node_dict.eos(), edge_dict.index("<blank>"),
                            stage)

    return data


def validation_acc(model, dev_iters, epoch, epochs, node_dict, edge_dict, max_nodes, cuda):
    """ Evaluate the model on dev set"""
    model.eval()
    eval_st = time.time()
    graphs, graph_corrects = 0, 0

    for i, dev_it in enumerate(dev_iters):
        if cuda:
            samples = move_to_cuda(dev_it)
        else:
            samples = dev_it

        _, _, _, batch_graph_correct = greedy_search(model, samples["src_graph"], samples["src_text"], samples["tgt_graph"],
                      node_dict, edge_dict, max_nodes, cuda)
        graph_corrects += batch_graph_correct
        graphs += 1

    acc = graph_corrects/graphs
    eval_time = (time.time() - eval_st) / 60
    eval_info = "[  Eval {:02}/{:02}]: accuracy={:.4f}  elapse={:.4f} mins"
    print(eval_info.format(epoch+1, epochs, acc, eval_time))

    model.train()
    return acc


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()

    node_dict, edge_dict, text_dict = load_dict(args)

    train_data = load_data(args, node_dict, edge_dict, text_dict)
    dev_data = load_data(args, node_dict, edge_dict, text_dict, stage="dev")

    train_tgt_sizes = [train_data.item_size(i, -1) for i in range(len(train_data))]
    dev_tgt_sizes = [dev_data.item_size(i, -1) for i in range(len(dev_data))]
    print(" [training]: {} examples".format(len(train_data)))
    print(" [dev     ]: {} examples".format(len(dev_data)))

    train_iters = DataLoader(train_data,
                             batch_sampler=BatchSampler(torch.tensor(train_tgt_sizes), batch=args.batch_size),
                             collate_fn=train_data.collate_fn)
    dev_iters = DataLoader(dev_data,
                           batch_sampler=BatchSampler(torch.tensor(dev_tgt_sizes), batch=1),
                           collate_fn=dev_data.collate_fn)

    model = GraphTrans(args, node_dict, edge_dict, text_dict)
    print('| num. model params: {} (num. trained: {})'.format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if cuda:
        model.cuda()
    opt = get_std_opt(args, model)
    print(model)

    # best_val = 9e+99
    best_val = 0.
    start_epoch = 0
    batch_step = 0
    opt.zero_grad()

    saved = load_model(args, model, optimizer=opt)

    # load save model and optimizer from disk
    if saved:
        best_val = saved["best_val"]
        # best_val = 0.
        start_epoch = saved["epoch"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_steps = 0., 0
        epoch_st = time.time()

        for train_it in train_iters:
            if cuda:
                samples = move_to_cuda(train_it)
            else:
                samples = train_it
            loss = model(samples["src_graph"], samples["src_text"], samples["tgt_graph"])
            loss = loss / args.accumulation_steps                # Normalize our loss (if averaged)
            epoch_loss += loss.item()
            loss.backward()
            if (batch_step + 1) % args.accumulation_steps == 0:             # Wait for several backward steps
                opt.clip_grad_norm(args.clip_norm)
                opt.step()
                opt.zero_grad()

                total_steps = opt.get_step()

                # evaluate the model on dev set
                if total_steps % args.eval_step == 0:
                    val_acc = validation_acc(model, dev_iters, epoch, args.epochs, node_dict, edge_dict, 10, cuda)
                    if val_acc > best_val:
                        save_model(args, model, opt, epoch, best_val, "best")
                        best_val = val_acc

                epoch_steps += 1
            batch_step += 1

        val_acc = validation_acc(model, dev_iters, epoch, args.epochs, node_dict, edge_dict, 10, cuda)
        if val_acc > best_val:
            save_model(args, model, opt, epoch, best_val, "best")
            best_val = val_acc

        save_model(args, model, opt, epoch+1, best_val, "last")
        epoch_time = (time.time() - epoch_st) / 60
        train_info = "[Train {:02}/{:02}]: total_loss={:.4f} avg_loss={:.4f} total_steps={:05} elapse={:.4f} mins best_val={:.4f} lr={:.4f}"
        print(train_info.format(epoch+1, args.epochs, epoch_loss, epoch_loss/epoch_steps, total_steps, epoch_time, best_val, opt.rate()))


if __name__ == "__main__":
    main()
