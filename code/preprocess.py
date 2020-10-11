#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : preprocess.py
from __future__ import print_function

import os
import sys

from data_utils import build_dictionary_from_bin, build_dictionary_from_text, build_dictionary


def main(data_dir, freq=3):
    # paths to graphs and queries
    graph_paths = [os.path.join(data_dir, "train_src_graph.bin"),
                   os.path.join(data_dir, "train_tgt_graph.bin")]
    text_path = os.path.join(data_dir, "train_src_text.txt")

    dict_path = os.path.join(data_dir, "dict.txt")
    build_dictionary(graph_paths, text_path, dict_path, freq)


if __name__ == "__main__":
    main(*sys.argv[1:])
