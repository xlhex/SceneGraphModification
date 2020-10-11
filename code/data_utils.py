#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : data_utils.py
from __future__ import print_function

import os
import pickle

import numpy as np
import torch

from collections import Counter

from torch.utils.data import Dataset, Sampler, DataLoader
from torch.nn import functional as F


def collate_tokens(values, pad_idx, shape):
    """Convert a list of nd tensors into a padded (n+1)d tensor."""
    res = values[0].new(*shape).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        if v.dim() == 1:
            copy_tensor(v, res[i][:len(v)])
        else:
            copy_tensor(v, res[i, :v.size(0), :v.size(1)])
    return res


def flatten_edge(edges):
    flat_edges = [edge.view(-1)[torch.tril(edge, -1).view(-1).nonzero()].view(-1) for edge in edges]

    return flat_edges


def shift_for_output(nodes, edges, eos_idx, pad_idx):
    # nodes
    nodes_x = [torch.cat([n.new(1).fill_(eos_idx), n]) for n in nodes]
    nodes_y = [torch.cat([n, n.new(1).fill_(eos_idx)]) for n in nodes]

    # edges
    edges_x = [torch.cat([e.new(1).fill_(eos_idx), e]) for e in edges]
    edges_y = [torch.cat([e, e.new(1).fill_(eos_idx)]) for e in edges]

    return nodes_x, nodes_y, edges_x, edges_y


def collate_fn(samples, pad_idx, eos_idx, train):

    def src_graph_batch(values):
        nodes = [s[0] for s in values]
        edges = [s[1] for s in values]

        bsz = len(nodes)
        max_size = max([n.size(0) for n in nodes])

        nodes_t = collate_tokens(nodes, pad_idx, (bsz, max_size))
        edges_t = collate_tokens(edges, pad_idx, (bsz, max_size, max_size))

        return nodes_t, edges_t

    def text_batch(values):
        tensors = [s for s in values]

        bsz = len(tensors)
        max_size = max([t.size(0) for t in tensors])

        tensor = collate_tokens(tensors, pad_idx, (bsz, max_size))

        return tensor

    def tgt_graph_batch(values):

        nodes = [s[0] for s in values]
        edges = [s[1] for s in values]

        edges = flatten_edge(edges)
        nodes_x, nodes_y, edges_x, edges_y = shift_for_output(nodes, edges, eos_idx, pad_idx)

        bsz = len(nodes)

        max_size = max([n.size(0) for n in nodes_x])
        nodes_x_t = collate_tokens(nodes_x, pad_idx, (bsz, max_size))
        nodes_y_t = collate_tokens(nodes_y, pad_idx, (bsz, max_size))

        max_size = max([e.size(0) for e in edges_x])
        edges_x_t = collate_tokens(edges_x, pad_idx, (bsz, max_size))
        edges_y_t = collate_tokens(edges_y, pad_idx, (bsz, max_size))

        return nodes_x_t, nodes_y_t, edges_x_t, edges_y_t

    ids = torch.LongTensor([s[0] for s in samples])
    src_nodes, src_edges = src_graph_batch([s[1] for s in samples])
    queries = text_batch([s[2] for s in samples])
    tgt_nodes_x, tgt_nodes_y, tgt_edges_x, tgt_edges_y = tgt_graph_batch([s[3] for s in samples])

    return {
            "ids": ids,
            "src_graph": {
                            "nodes": src_nodes,
                            "edges": src_edges
                         },
            "src_text": {"x": queries},
            "tgt_graph": {
                            "nodes": {
                                        "x": tgt_nodes_x,
                                        "y": tgt_nodes_y
                                     },
                             "edges": {
                                        "x": tgt_edges_x,
                                        "y": tgt_edges_y
                                     }
                         }
            }


class Dictionary(object):
    def __init__(self, pad="<pad>", eos="</s>", unk="<unk>"):
        self.pad_word = pad
        self.eos_word = eos
        self.unk_word = unk

        self.symbols = []
        self.indices = {}

        self.pad_index = self.add_symbol(pad)
        self.unk_index = self.add_symbol(unk)
        self.eos_index = self.add_symbol(eos)

        self.n_specials = len(self.symbols)

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        return len(self.symbols)

    def index(self, sym):
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_word
            else:
                return self[i]

        sent = ' '.join(token_string(i) for i in tensor)
        return sent

    def add_symbol(self, word):
        if word in self.indices:
            idx = self.indices[word]
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)

        return idx

    def pad(self):
        return self.pad_index

    def unk(self):
        return self.unk_index

    def eos(self):
        return self.eos_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:
        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        with open(f) as reader:
            for line in reader:
                line = line.strip()
                d.add_symbol(line)
        return d

    def save(self, dict_file):
        if not os.path.exists(dict_file):
            with open(dict_file, "w") as f:
                for sym in self.symbols[self.n_specials:]:
                    f.write(sym)
                    f.write("\n")

    def encode_line(self, line, append_eos=True):
        if isinstance(line, str):
            words = line.strip().split()
        else:
            words = line
        nwords = len(words)

        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            ids[i] = self.index(word)

        if append_eos:
            ids[nwords] = self.eos_index

        return ids


class GraphTransReader(Dataset):
    """
    Graph modification dataset
    """
    def __init__(self, src_graph, query, tgt_graph, pad_idx, eos_idx, blank_idx, stage="train"):
        assert len(src_graph) == len(query) == len(tgt_graph)

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.blank_idx = blank_idx
        self.stage = stage

        self.src_graph = src_graph
        self.query = query
        self.tgt_graph = tgt_graph
        self.size = len(src_graph)
        self.sizes = [src_graph.sizes, query.sizes, tgt_graph.sizes]

    def __getitem__(self, i):
        return i, self.src_graph[i], self.query[i], self.tgt_graph[i]

    def __len__(self):
        return self.size

    def item_size(self, index, type=0):
        return self.sizes[type][index]

    def collate_fn(self, samples):
        return collate_fn(samples, self.pad_idx, self.eos_idx, train=self.stage=="train")


class BatchSampler(Sampler):
    def __init__(self, sizes, batch=32):
        _, indices = torch.sort(sizes, descending=True)
        s = []
        for i in range(0, len(indices), batch):
            s.append(indices[i:i+batch])

        self.indices = s

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class GraphReader(Dataset):
    """
    Graph dataset
    """
    def __init__(self, path, node_dictionary, edge_dictionary):
        self.nodes_list = []
        self.edges_list = []
        self.graphs = []
        self.sizes = []
        self.read_data(path, node_dictionary, edge_dictionary)
        self.size = len(self.nodes_list)

    def read_data(self, path, node_dict, edge_dict):
        def parse_graph(g, node_dict, edge_dict):
            """
            convert graph stored in networkx into nodes and edges
            """
            num_nodes = g.number_of_nodes()
            nodes = [node[-1]["feature"] for node in g.nodes.data()]
            edges = [["<blank>" for _ in range(num_nodes)] for _ in range(num_nodes)]
            for edge in g.edges.data():
                edges[edge[0]][edge[1]] = edge[2]["feature"]
                edges[edge[1]][edge[0]] = edge[2]["feature"]

            nodes = node_dict.encode_line(nodes, False).long()
            edges = torch.stack([edge_dict.encode_line(e, False) for e in edges], dim=0).to(nodes.dtype)
            return nodes, edges

        def load_data(output_file):
            with open(output_file, "rb") as fr:
                while True:
                    try:
                        yield pickle.load(fr)
                    except EOFError:
                        break

        graphs = load_data(path)
        for i, graph in enumerate(graphs):
            # if i > 1000: break
            self.graphs.append(graph)
            nodes, edges = parse_graph(graph, node_dict, edge_dict)
            self.nodes_list.append(nodes)
            self.edges_list.append(edges)
            self.sizes.append(nodes.size(0))

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, i):
        self.check_index(i)
        return self.nodes_list[i], self.edges_list[i]

    def __len__(self):
        return self.size

    def item_size(self, index):
        return self.sizes[index]


class TextReader(Dataset):
    """
    Query dataset
    """
    def __init__(self, path, dictionary):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # if i > 1000: break
                self.lines.append(line.strip())
                tokens = dictionary.encode_line(line).long()
                self.tokens_list.append(tokens)
                self.sizes.append(tokens.size(0))

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def __len__(self):
        return self.size

    def item_size(self, index):
        return self.sizes[index]


def build_dictionary_from_text(data_file, dict_file, threshold=0, max_vocab=-1):
    if not os.path.exists(dict_file):
        c = Counter()
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                c.update(line)

        if max_vocab != -1:
            vocab = [tok for tok, _ in c.most_common(max_vocab)]
        else:
            vocab = [tok for tok, count in c.items() if count >= threshold]

        with open(dict_file, "w") as f:
            for v in vocab:
                f.write(v)
                f.write("\n")


def build_dictionary_from_bin(data_files, node_dict_file, edge_dict_file, threshold=0, max_vocab=-1):
    if not os.path.exists(node_dict_file) or not os.path.exists(edge_dict_file):
        def load_data(output_file):
            with open(output_file, "rb") as fr:
                while True:
                    try:
                        yield pickle.load(fr)
                    except EOFError:
                        break

        c_node = Counter()
        c_edge = Counter()
        for data_file in data_files:
            graphs = load_data(data_file)
            for graph in graphs:
                c_node.update([node[-1]["feature"] for node in graph.nodes.data()])
                c_edge.update([edge[-1]["feature"] for edge in graph.edges.data()])

        if max_vocab != -1:
            node_vocab = [node for node, _ in c_node.most_common(max_vocab)]
            edge_vocab = [edge for edge, _ in c_edge.most_common(max_vocab)]
        else:
            node_vocab = [node for node, count in c_node.items() if count >= threshold]
            edge_vocab = [edge for edge, count in c_edge.items() if count >= threshold]

        with open(node_dict_file, "w") as f:
            for v in node_vocab:
                f.write(v)
                f.write("\n")

        with open(edge_dict_file, "w") as f:
            for v in edge_vocab:
                f.write(v)
                f.write("\n")


def build_dictionary(graph_data_files, text_data_file, dict_file, threshold=0, max_vocab=-1):
    if not os.path.exists(dict_file):
        def load_data(output_file):
            with open(output_file, "rb") as fr:
                while True:
                    try:
                        yield pickle.load(fr)
                    except EOFError:
                        break

        tokens = Counter()

        for data_file in graph_data_files:
            graphs = load_data(data_file)
            for graph in graphs:
                tokens.update([node[-1]["feature"] for node in graph.nodes.data()])
                tokens.update([edge[-1]["feature"] for edge in graph.edges.data()])

        with open(text_data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()
                tokens.update(line)

        if max_vocab != -1:
            vocab = [node for node, _ in tokens.most_common(max_vocab)]
        else:
            vocab = [node for node, count in tokens.items() if count >= threshold]

        with open(dict_file, "w") as f:
            for v in vocab:
                f.write(v)
                f.write("\n")
