# Scene Graph Modification Based on Natural Language Commands

## Descriptions
This repo contains source code and pre-processed corpora for __"Scene Graph Modification Based on Natural Language Commands"__ (accepted to Findings of EMNLP 2020) ([paper](https://arxiv.org/abs/2010.02591))


## Dependencies
* python3
* pytorch>=1.4
* networkx
* cuda 10.0

## Usage
```shell
https://github.com/xlhex/SceneGraphModification.git
```

## Data
### General Information
We create three different datasets for our scene graph modification task: 1) MSCOCO data, 2) GCC data and 3) crowdsourced data. The first two are constructed with some heuristic approaches, while the last one is crowdsourced from Amazon Mechanical Turk (please refer to ourt paper for the details).

Each dataset is partitioned into train/dev/test, with each split consisting of the following files:
* source scene graph: {split}_src_graph.bin
* modification query: {split}_src_text.txt
* target scene graph: {split}_dev_graph.bin
* MSCOCO data: data/mscoco_data
* GCC data: data/GCC_data
* Crowdsourced data: data/crowdsourced_data

The datasets can be downloaded from [here](https://drive.google.com/file/d/1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY/view?usp=sharing)
