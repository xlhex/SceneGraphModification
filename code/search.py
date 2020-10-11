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

from collections import Counter
from functools import reduce

from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils import get_parser, save_model, load_model, get_std_opt, move_to_cuda
from models import GraphTrans
from data_utils import Dictionary, GraphReader, TextReader, GraphTransReader, BatchSampler


def load_dict(args):
    node_dict = Dictionary().load(os.path.join(args.data_dir, "dict.txt"))

    node_dict.add_symbol("<blank>")
    edge_dict = node_dict
    text_dict = node_dict

    return node_dict, edge_dict, text_dict


def load_data(args, node_dict, edge_dict, text_dict, stage="train"):
    src_graph = GraphReader(os.path.join(args.data_dir, "{}_src_graph.bin".format(stage)), node_dict, edge_dict)
    src_text = TextReader(os.path.join(args.data_dir, "{}_src_text.txt".format(stage)), text_dict)
    tgt_graph = GraphReader(os.path.join(args.data_dir, "{}_tgt_graph.bin".format(stage)), node_dict, edge_dict)
    data = GraphTransReader(src_graph, src_text, tgt_graph,
                            node_dict.pad(), node_dict.eos(), edge_dict.index("<blank>"),
                            stage)

    return data


def decoding(graph_dec, enc_info, node_dict, edge_dict, max_nodes, cuda):
    step = 0

    input = torch.tensor([[node_dict.eos()]])
    inputs = []
    nodes_len = torch.tensor([1])
    if cuda:
        input = input.cuda()
        nodes_len = nodes_len.cuda()

    h_prev = None
    h_list = []

    while step < max_nodes:
        _, h_cur, logits = graph_dec.node_forward(enc_info, input, nodes_len, h_prev)

        if step == 0:
            logits[:, :, node_dict.eos()] = -1e9
        next_input = F.softmax(logits, dim=-1).argmax(dim=-1)
        input = next_input
        h_prev = h_cur

        new_tok = next_input.view(-1).item()
        if new_tok == node_dict.eos():
            break
        inputs.append(new_tok)
        h_list.append(h_cur)

        step += 1

    node_rnn_outputs = torch.cat(h_list, dim=1)
    max_step = len(h_list)
    edges = []
    h_prev = None
    input = torch.tensor([[edge_dict.eos()]])
    if cuda:
        input = input.cuda()

    src_nodes = reduce(lambda x, y: x+y, [[i for _ in range(i)] for i in range(1, max_step)]) if max_step > 1 else []
    tgt_nodes = reduce(lambda x, y: x+y, [[j for j in range(i)] for i in range(1, max_step)]) if max_step > 1 else []

    edge_rnn_outputs = []
    for src, tgt in zip(src_nodes, tgt_nodes):
        src_node_states = node_rnn_outputs[:, src:src+1]
        tgt_node_states = node_rnn_outputs[:, tgt:tgt+1]
        _, h_cur, logits = graph_dec.edge_forward(enc_info, input, src_node_states, tgt_node_states, h_prev)
        logits[:, :, edge_dict.eos()] = -1e9
        next_input = F.softmax(logits, dim=-1).argmax(dim=-1)
        input = next_input
        edges.append(next_input.view(-1).cpu().item())
        h_prev = h_cur
        edge_rnn_outputs.append(h_prev)

    if edge_rnn_outputs:
        edge_rnn_outputs = torch.cat(edge_rnn_outputs, dim=1)
    else:
        bsz = node_rnn_outputs.size(0)
        edge_rnn_outputs = node_rnn_outputs.new_zeros(bsz, 0, graph_dec.args.edge_hidden_size)

    return inputs, node_rnn_outputs, edges, edge_rnn_outputs, src_nodes, tgt_nodes


def greedy_search(model, src_graph, src_text, tgt_graph,
                  node_dict, edge_dict, max_nodes, cuda):
    # graph encoder
    enc_info = model.encoder(src_graph, src_text)
    inputs, _, edges, _, src_nodes, tgt_nodes = decoding(model.graph_dec, enc_info,
                                                         node_dict, edge_dict, max_nodes, cuda)
        
    node_c = Counter()
    act_outputs = [i.item() for i in tgt_graph["nodes"]["y"][0][:-1]]
    node_total_num = len(act_outputs)
    node_c.update(act_outputs)
    for node in inputs:
        # if node in node_c and node_c[node] > 0:
        if node in node_c and node_c[node] > 0 and node != node_dict.unk():
            node_c[node] -= 1

    node_incorrect = sum([node_c[k] for k in node_c])
    # print(node_total_num, node_incorrect)
    #print(node_dict.string(inputs))
    #print(node_dict.string(tgt_graph["nodes"]["y"][0][:-1]))
    # print("*"*20)
    # print(edge_dict.string(adj))
    pred_edges = []
    act_edges = []
    edge_c = Counter()
    edge_total_num = 0
    ref_src_nodes = reduce(lambda x,y:x+y, [[i for _ in range(i)] for i in range(1, node_total_num)]) if node_total_num > 1 else []
    ref_tgt_nodes = reduce(lambda x,y:x+y, [[j for j in range(i)] for i in range(1, node_total_num)]) if node_total_num > 1 else []
    ref_edges = tgt_graph["edges"]["y"][0][:-1].cpu()
    # ground truth
    for edge, src, tgt in zip(ref_edges, ref_src_nodes, ref_tgt_nodes):
        if edge != edge_dict.index("<blank>"):
            # arc = tuple(sorted([node_dict[act_outputs[src]], edge_dict[edge], node_dict[act_outputs[tgt]]]))
            arc = tuple(sorted([node_dict[act_outputs[src]] if act_outputs[src] != node_dict.unk() else "<<unk>>", edge_dict[edge] if edge != edge_dict.unk() else "<<unk>>", node_dict[act_outputs[tgt]] if act_outputs[tgt] != node_dict.unk() else "<<unk>>"]))
            #arc = (node_dict[act_outpus[src]], edge_dict[edge], node_dict[act_outpus[tgt]])
            edge_c[arc] += 1
            edge_total_num += 1
            act_edges.append(arc)
    # predicted edges
    for edge, src, tgt in zip(edges, src_nodes, tgt_nodes):
        if edge != edge_dict.index("<blank>"):
            arc = tuple(sorted([node_dict[inputs[src]], edge_dict[edge], node_dict[inputs[tgt]]]))
            #arc = (node_dict[inputs[src]], edge_dict[edge], node_dict[inputs[tgt]])
            pred_edges.append(arc)
            if arc in edge_c and edge_c[arc] > 0:
                edge_c[arc] -= 1
    edge_incorrect = sum([edge_c[k] for k in edge_c])
    # print(edge_total_num, edge_incorrect)
    # print("-"*20)
    # total_num = node_total_num + edge_total_num
    # incorrect = node_incorrect + edge_incorrect
    # total_pred = len(inputs) + len(pred_edges)
    # total_num = edge_total_num
    # incorrect = edge_incorrect
    # total_pred = len(pred_edges)
    node_correct = node_total_num - node_incorrect
    edge_correct = edge_total_num - edge_incorrect
    graph_correct = 0

    if sorted(inputs) == sorted(act_outputs) and sorted(pred_edges) == sorted(act_edges):
        graph_correct = 1

    return (node_correct, edge_correct), (node_total_num, edge_total_num), (len(inputs), len(pred_edges)), graph_correct


def main():
    parser = get_parser("test")
    args = parser.parse_args()

    print(args)

    cuda = torch.cuda.is_available()

    node_dict, edge_dict, text_dict = load_dict(args)

    test_data = load_data(args, node_dict, edge_dict, text_dict,
                          stage="test")

    test_tgt_sizes = [test_data.item_size(i, -1) for i in range(len(test_data))]
    print(" [test]: {} examples".format(len(test_data)))

    test_iters = DataLoader(test_data,
                            batch_sampler=BatchSampler(torch.tensor(test_tgt_sizes), batch=args.batch_size),
                            collate_fn=test_data.collate_fn)

    model = GraphTrans(args, node_dict, edge_dict, text_dict)
    model.eval()
    if cuda:
        model.cuda()

    saved = load_model(args, model, inference=True)
    if not saved:
        raise FileNotFoundError("Checkpoint does not exist")

    edges_correct, edges_num, edges_pred = 0, 0, 0
    nodes_correct, nodes_num, nodes_pred = 0, 0, 0
    graphs, graph_corrects = 0, 0

    for i, test_it in enumerate(test_iters):
        if cuda:
            samples = move_to_cuda(test_it)
        else:
            samples = test_it

        batch_correct, batch_num, batch_pred, batch_graph_correct = greedy_search(model, samples["src_graph"], samples["src_text"], samples["tgt_graph"],
                      node_dict, edge_dict, args.max_nodes, cuda, False)

        nodes_correct += batch_correct[0]
        nodes_num += batch_num[0]
        nodes_pred += batch_pred[0]
        edges_correct += batch_correct[1]
        edges_num += batch_num[1]
        edges_pred += batch_pred[1]
        graph_corrects += batch_graph_correct
        graphs += 1

    print("Node: Recall: {:.2f}({}/{}), Precision: {:.2f}({}/{}) ".format(nodes_correct/nodes_num * 100, nodes_correct, nodes_num, nodes_correct/nodes_pred * 100, nodes_correct, nodes_pred))
    print("Edge: Recall: {:.2f}({}/{}), Precision: {:.2f}({}/{}) ".format(edges_correct/edges_num * 100, edges_correct, edges_num, edges_correct/edges_pred * 100, edges_correct, edges_pred))
    print("Accuracy: {:.2f}({}/{})".format(graph_corrects/graphs * 100, graph_corrects, graphs))


if __name__ == "__main__":
    main()
