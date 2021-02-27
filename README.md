# Scene Graph Modification Based on Natural Language Commands

## Descriptions
This repo contains source code and pre-processed corpora for __"Scene Graph Modification Based on Natural Language Commands"__ (accepted to Findings of EMNLP 2020) ([paper](https://arxiv.org/abs/2010.02591))

## Demo
We demonstrate four different operations one can execute on scene graphs:
![](https://github.com/xlhex/SceneGraphModification/blob/master/demo/graphtrans_demo1.gif)
![](https://github.com/xlhex/SceneGraphModification/blob/master/demo/graphtrans_demo2.gif)


## Dependencies
* python3
* pytorch==1.1
* networkx
* spacy>=2.3.1

## Usage
```shell
git clone https://github.com/xlhex/SceneGraphModification.git
```

## Data
### General Information
We create three different datasets for our scene graph modification task: 1) MSCOCO data, 2) GCC data and 3) crowdsourced data. The first two are constructed with some heuristic approaches, while the last one is crowdsourced from Amazon Mechanical Turk (please refer to our paper for the details).

Each dataset is partitioned into train/dev/test, with each split consisting of the following files:
* source scene graph: {split}_src_graph.bin
* modification query: {split}_src_text.txt
* target scene graph: {split}_tgt_graph.bin

The datasets can be downloaded from [here](https://drive.google.com/file/d/1K2lo1Dt7GJskyUVR9x5LH-mZya28KcDY/view?usp=sharing)

## Train a model
The following code shows how we can train an early fusion (cross-attention) model for a given dataset
```shell
cd code

DATA=PATH_TO_YOUR_DATA
CKPT_DIR=
EPOCH=20
FUSION=early

log="${CKPT_DIR}/log.txt"
if [ ! -d $CKPT_DIR ];then
    mkdir -p $CKPT_DIR
fi

# build a dictionary for training and inference
python preprocess.py $DATA

python train.py --data-dir $DATA --epochs $EPOCH --seed 1 --ckpt-dir $CKPT_DIR --modification $FUSION --batch-size 256 --accumulation-steps 1 > $log
```

## Inference
The following code shows how we generate a target graph, given the source graph and a modification query
```shell
cd code

DATA=PATH_TO_YOUR_DATA
CKPT_DIR=
FUSION=early

python search.py --data-dir $DATA --greedy-search --batch-size 1 --ckpt-dir $CKPT_DIR --modification $FUSION
```

## Instance Visualisation
You can visualise some modification instances. For example, the following code will visualise the first two instances. The rendered source graphs and target graphs can be found at: `scripts/display`
```shell

cd scripts
SRC_GRAPH=PATH_TO_SRC_GRAPH
TGT_GRAPH=PATH_TO_TGT_GRAPH
QUERY=PATH_TO_QUERY

python visualisation.py --src-graph $SRC_GRAPH --tgt-graph $TGT_GRAPH --graph-idx 0,1 --query $QUERY
```

## Citation

Please cite as:

```bibtex
@misc{he2020scene,
      title={Scene Graph Modification Based on Natural Language Commands}, 
      author={Xuanli He and Quan Hung Tran and Gholamreza Haffari and Walter Chang and Trung Bui and Zhe Lin and Franck Dernoncourt and Nhan Dam},
      year={2020},
      eprint={2010.02591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
