# graph-rcnn.pytorch
Pytorch code for our ECCV 2018 paper ["Graph R-CNN for Scene Graph Generation"](https://arxiv.org/pdf/1808.00191.pdf)

<div style="color:#0000FF" align="center">
<img src="figures/teaser_fig.png" width="850"/>
</div>

<!-- :balloon: 2019-06-04: Okaaay, time to reimplement Graph R-CNN on pytorch 1.0 and release a new benchmark for scene graph generation. It will also integrate other models like IMP, MSDN and Neural Motif Network. Stay tuned!

:balloon: 2019-06-16: Plan is a bit delayed by ICCV rebuttal, but still on track. Stay tuned! -->

## Introduction

This project is a set of reimplemented representative scene graph generation models based on Pytorch 1.0, including:
* [Graph R-CNN for Scene Graph Generation](https://arxiv.org/pdf/1808.00191.pdf), our own. ECCV 2018.
* [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/pdf/1701.02426.pdf), Xu et al. CVPR 2017
* [Scene Graph Generation from Objects, Phrases and Region Captions](https://arxiv.org/pdf/1707.09700.pdf), Li et al. ICCV 2017
* [Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/pdf/1711.06640.pdf), Zellers et al. CVPR 2018

Our reimplementations are based on the following repositories:

* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
* [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch)
* [scene-graph-TF-release](https://github.com/danfeiX/scene-graph-TF-release)
* [MSDN](https://github.com/yikang-li/MSDN)
* [neural-motifs](https://github.com/rowanz/neural-motifs)

## Why we need this repository?

The goal of gathering all these representative methods into a single repo is to establish a more fair comparison across different methods under the same settings. **As you may notice in recent literatures, the reported numbers for IMP, MSDN, Graph R-CNN and Neural Motifs are usually confusing, especially due to the big gap between IMP style methods (first three) and Neural Motifs-style methods (neural motifs paper and other variants built on it)**. We hope this repo can establish a good benchmark for various scene graph generation methods, and contribute to the research community!

## Checklist

- [x] Faster R-CNN Baseline (:balloon: 2019-07-04)
- [x] Scene Graph Generation Baseline (:balloon: 2019-07-06)
- [x] Iterative Message Passing (IMP) (:balloon: 2019-07-07)
- [ ] Multi-level Scene Description Network (MSDN)
- [x] Neural Motif (Frequency Prior Baseline) (:balloon: 2019-07-08)
- [ ] Neural Motif
- [ ] Graph R-CNN

## Benchmarking

### Object Detection

source  | backbone | model | bs | lr  | lr_decay | max_iter | mAP@0.5 | mAP@0.50:0.95
--------|--------|--------|--------|--------|---------|--------|--------|--------
this repo | Res-101 | faster r-cnn | 6 | 5e-3 | (70k, 90k) | 100k | 24.8 | 12.8

### Scene Graph Generation
source | backbone | model | bs | lr | lr_decay | max_iter | sgdet@20 | sgdet@50 | sgdet@100
-------|--------|--------|--------|---------|--------|--------|--------|---------|---------
this repo | Res-101 | vanilla | 6 | 5e-3 | (70k, 90k) | 100k | 10.4 | 14.3 | 16.8
this repo | Res-101 | freq | 6 | 5e-3 | (70k, 90k) | 100k | 19.4 | 25.0 | 28.5
[neural motif](https://github.com/rowanz/neural-motifs) | VGG-16 | freq | N/A | N/A | N/A | N/A | 17.7 | 23.5 | 27.6
<!-- Resnet-101 | freq-overlap | 6 | 5e-3 | (70k, 90k) | 100k | - | - | - -->
\* freq = frequency prior baseline


## Installation

### Prerequisites

* Python 3.6+
* Pytorch 1.0
* CUDA 8.0+

### Dependencies

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

### Data Preparation

* Visual Genome benchmarking dataset:

Annotations | Object | Predicate
------------|--------| ---------
\#Categories| 150    | 50

First, make a folder in the root folder:
```
mkdir -p datasets/vg_bm
```

Here, the suffix 'bm' is in short of "benchmark" representing the dataset for benchmarking. We may have other format of vg dataset in the future, e.g., more categories.

Then, download the data and preprocess the data according following this [repo](https://github.com/danfeiX/scene-graph-TF-release). Specifically, after downloading  the [visual genome dataset](https://visualgenome.org/), you can follow this [guidelines](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools) to get the following files:

```
datasets/vg_bm/imdb_1024.h5
datasets/vg_bm/bbox_distribution.npy
datasets/vg_bm/proposals.h5
datasets/vg_bm/VG-SGG-dicts.json
datasets/vg_bm/VG-SGG.h5
```

The above files will provide all the data needed for training the object detection models and scene graph generation models listed above.

* Visual Genome bottom-up and top-down dataset:

Annotations | Object | Attribute | Predicate
------------|--------|-----------|-----------
\#Categories| 1600   | 400       | 20

Soon, I will add this data loader to train [bottom-up and top-down model](https://arxiv.org/pdf/1707.07998.pdf) on more object/predicate/attribute categories.

* Visual Genome extreme dataset:

Annotations | Object | Attribute | Predicate
------------|--------|-----------|-----------
\#Categories| 2500   | ~600      | ~400

This data loader further increase the number of categories for training more fine-grained visual representations.

### Compilation

Compile the cuda dependencies using the following commands:
```
cd lib/scene_parser/rcnn
python setup.py build develop
```

After that, you should see all the necessary components, including nms, roi_pool, roi_align are compiled successfully.

## Train

### Train object detection model:

* Faster r-cnn model with resnet-101 as backbone:
```
python main.py --config-file configs/faster_rcnn_res101.yaml
```

Multi-GPU training:
```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/faster_rcnn_res101.yaml
```
where NGPUS is the number of gpus available.

### Train scene graph generation model:

* Vanilla scene graph generation model with resnet-101 as backbone:
```
python main.py --config-file configs/baseline_res101.yaml
```

Multi-GPU training:
```
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py --config-file configs/baseline_res101.yaml
```
where NGPUS is the number of gpus available.

## Evaluate

### Evaluate object detection model:

* Faster r-cnn model with resnet-101 as backbone:
```
python main.py --config-file configs/faster_rcnn_res101.yaml --inference --resume $CHECKPOINT
```
where CHECKPOINT is the iteration number. By default it will evaluate the whole validation/test set. However, you can specify the number of inference images by appending the following argument:
```
--inference $YOUR_NUMBER
```

### Evaluate scene graph generation model:

* Vanilla scene graph generation model with resnet-101 as backbone:
```
python main.py --config-file configs/baseline_res101.yaml --inference --resume $CHECKPOINT
```

* Vanilla scene graph generation model with resnet-101 as backbone and use frequency prior:
```
python main.py --config-file configs/baseline_res101.yaml --inference --resume $CHECKPOINT --use_freq_prior
```

Similarly you can also append the ''--inference $YOUR_NUMBER'' to perform partially evaluate.
