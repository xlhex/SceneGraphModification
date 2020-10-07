#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : preprocess.py
from __future__ import print_function

import os
import sys

from data_utils import build_dictionary_from_bin, build_dictionary_from_text, build_dictionary

def main(data_dir):
    graph_paths = [os.path.join(data_dir, "train_src_graph.bin"),
                   os.path.join(data_dir, "train_tgt_graph.bin")]
    text_path = os.path.join(data_dir, "train_src_text.txt")
    # node_dict_path = os.path.join(data_dir, "node_dict.txt")
    # edge_dict_path = os.path.join(data_dir, "edge_dict.txt")
    # text_dict_path = os.path.join(data_dir, "text_dict.txt")

    # build_dictionary_from_text(text_path, text_dict_path, 2)
    # build_dictionary_from_bin(graph_paths, node_dict_path, edge_dict_path, 2)
    dict_path = os.path.join(data_dir, "dict.txt")
    build_dictionary(graph_paths, text_path, dict_path, 3)

if __name__ == "__main__":
    main(sys.argv[1])
