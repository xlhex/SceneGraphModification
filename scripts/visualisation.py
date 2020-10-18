#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : visualisation.py
from __future__ import print_function

import argparse
import os
import pickle
import sys
import random
from collections import Counter

from spacy import displacy

def load(output_file):
    with open(output_file, "rb") as fr:
        while True:
            try:
                yield pickle.load(fr)
            except EOFError:
                break


def main(args):
    items = load(args.src_graph)
    graph_src = []
    for i, item in enumerate(items):
        nodes = []
        edges = []
        for n in item.nodes.data():
            nodes.append(n[1]["feature"])
        for e in item.edges.data():
            edges.append((e[0], e[2]["feature"], e[1]))
        
        graph_src.append((nodes, edges))

    items = load(args.tgt_graph)
    graph_tgt = []
    for i, item in enumerate(items):
        nodes = []
        edges = []
        # if i > 4: break
        for n in item.nodes.data():
            nodes.append(n[1]["feature"])
        for e in item.edges.data():
            edges.append((e[0], e[2]["feature"], e[1]))
            # edges.append(e[2]["feature"])
        
        graph_tgt.append((nodes, edges))

    queries = []

    with open(args.query) as f:
        for line in f:
            queries.append(line.strip())

    interested_idx = [int(i) for i in args.graph_idx.split(",")]

    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    for i in interested_idx:
        src_nodes = graph_src[i][0]
        src_edges = graph_src[i][1]
        tgt_nodes = graph_tgt[i][0]
        tgt_edges = graph_tgt[i][1]
        query = queries[i]

        docs = []

        words = [{"text": node, "tag": ""} for node in src_nodes]
        arcs = [{"start": arc[0], "end": arc[2], "label": arc[1], "dir": "left" if arc[1] == "att" else "right"} for arc in src_edges]
        docs.append({"words": words, "arcs": arcs})

        words = [{"text": node, "tag": ""} for node in tgt_nodes]
        arcs = [{"start": arc[0], "end": arc[2], "label": arc[1], "dir": "left" if arc[1] == "att" else "right"} for arc in tgt_edges]
        docs.append({"words": words, "arcs": arcs})

        # output souce graph
        svg = displacy.render(docs[0], style="dep", jupyter=False, manual=True)
        output_path = os.path.join(args.dump_dir, "{:06}_src.svg".format(i))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)

        # output target graph
        svg = displacy.render(docs[1], style="dep", jupyter=False, manual=True)
        output_path = os.path.join(args.dump_dir, "{:06}_tgt.svg".format(i))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)

        # output modification query
        print("id {}: {}".format(i, query))


if __name__ == "__main__":
    """Parsing arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-graph", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--tgt-graph", required=True)
    parser.add_argument("--graph-idx", default="0", help="graphs of interest in a format of comma separated list, e.g. 1,2,3")
    parser.add_argument("--dump-dir", default="display")

    args = parser.parse_args()

    main(args)
