# CLIP4Cir (ACM TOMM 2023)

### Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.11485)
[![GitHub Stars](https://img.shields.io/github/stars/ABaldrati/CLIP4Cir?style=social)](https://github.com/ABaldrati/CLIP4Cir)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/composed-image-retrieval-using-contrastive/image-retrieval-on-fashion-iq)](https://paperswithcode.com/sota/image-retrieval-on-fashion-iq?p=composed-image-retrieval-using-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/composed-image-retrieval-using-contrastive/image-retrieval-on-cirr)](https://paperswithcode.com/sota/image-retrieval-on-cirr?p=composed-image-retrieval-using-contrastive)

This is the **official repository** for the [**paper**](https://arxiv.org/abs/2308.11485) "*Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features*".


## Citation
If you make use of our work, please cite our paper:
```bibtex
@article{baldrati2023composed,
  title={Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features},
  author={Baldrati, Alberto and Bertini, Marco and Uricchio, Tiberio and Bimbo, Alberto Del},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  publisher={ACM New York, NY}
}
```

If you are interested in Composed Image Retrieval (CIR) take a look also a look to our most recent work:
[**Zero-Shot Composed Image Retrieval with Textual Inversion (ICCV 2023)**](https://arxiv.org/abs/2303.15247)
[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/miccunifi/SEARLE)
```bibtex
@misc{baldrati2023zeroshot,
      title={Zero-Shot Composed Image Retrieval with Textual Inversion}, 
      author={Alberto Baldrati and Lorenzo Agnolucci and Marco Bertini and Alberto Del Bimbo},
      year={2023},
      eprint={2303.15247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## About The Project

### Abstract

Given a query composed of a reference image and a relative caption, the Composed Image Retrieval goal is to retrieve 
images visually similar to the reference one that integrates the modifications expressed by the caption. Given that 
recent research has demonstrated the efficacy of large-scale vision and language pretrained (VLP) models in various 
tasks, we rely on features from the OpenAI CLIP model to tackle the considered task. We initially perform a task-oriented 
fine-tuning of both CLIP encoders using the element-wise sum of visual and textual features. Then, in the second stage, 
we train a Combiner network that learns to combine the image-text features integrating the bimodal information and 
providing combined features used to perform the retrieval. We use contrastive learning in both stages of training. 
Starting from the bare CLIP features as a baseline, experimental results show that the task-oriented fine-tuning and 
the carefully crafted Combiner network are highly effective and outperform more complex state-of-the-art approaches on 
FashionIQ and CIRR, two popular and challenging datasets for composed image retrieval.


### Composed image retrieval task

![](images/cir-overview.png "Composed image retrieval overview")

The left portion of the illustration depicts a specific case of composed image retrieval in the fashion domain, where
the user imposes constraints on the character attribute of a t-shirt. Meanwhile, the right part showcases an example 
where the user asks to alter objects and their cardinality within a real-life image.
### CLIP task-oriented fine-tuning 

![](images/clip-fine-tuning.png "CLIP task oriented fine-tuning")

First stage of training. In this stage, we perform a task-oriented fine-tuning of CLIP encoders to reduce the mismatch 
between the large-scale pre-training and the downstream task. We start by extracting the 
 image-text query features and combining them through an element-wise sum. We then employ a contrastive loss 
to minimize the distance between combined features and target image features in the same triplet and maximize the 
distance from the other images in the batch. We update the weights of both CLIP encoders.
### Combiner training 

![](images/combiner-training.png "Combiner training")

Second stage of training. In this stage, we train from scratch a Combiner network that learns to fuse the multimodal 
features extracted with CLIP encoders. We start by extracting the image-text query features using the fine-tuned 
encoders, and we combine them using the Combiner network. We then employ a contrastive loss to minimize the distance 
between combined features and target image features in the same triplet and maximize the distance from the other images 
in the batch. We keep both CLIP encoders frozen while we only update the weights of the Combiner network.
At inference time the fine-tuned encoders and the trained Combiner are used to produce an effective representation used 
to query the database.

### Combiner architecture

![](images/Combiner-architecture.png "Combiner architecture overview")

Architecture of the Combiner network $C_{\theta}$. It takes as input the multimodal query features and outputs a unified
representation. $\sigma$ represents the sigmoid function. We denote the outputs of the first branch (1) as $\lambda$ 
and $1 -\lambda$, while the output of the second branch (2) as $v$. The combined features are $\overline{\phi_q} = (1 - \lambda)* \overline{\psi_{I}}(I_q) + \lambda * \overline{\psi_{T}}(T_q) + v$

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

We strongly recommend the use of the [**Anaconda**](https://www.anaconda.com/) package manager to avoid
dependency/reproducibility problems.
A conda installation guide for Linux systems can be
found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

### Installation

1. Clone the repo

```sh
git clone https://github.com/ABaldrati/CLIP4Cir
```

2. Install Python dependencies

```sh
conda create -n clip4cir -y python=3.8
conda activate clip4cir
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
conda install -y -c anaconda pandas=1.4.2
pip install comet-ml==3.21.0
pip install git+https://github.com/openai/CLIP.git
```

## Usage

Here's a brief description of each file under the ```src/``` directory:

For running the following scripts in a decent amount of time, it is **heavily** recommended to use a CUDA-capable GPU.
It is also recommended to have a properly initialized Comet.ml account to have better logging of the metrics
(all the metrics will also be logged on a csv file).

* ```utils.py```: utils file
* ```combiner.py```: Combiner model definition
* ```data_utils.py```: dataset loading and preprocessing utils
* ```clip_fine_tune.py```: CLIP task-oriented fine-tuning file
* ```combiner_train.py```: Combiner training file
* ```validate.py```: compute metrics on the validation sets
* ```cirr_test_submission.py```: generate test prediction on cirr test set

**N.B** The purpose of the code in this repo is to be as clear as possible. For this reason, it does not include some optimizations such as gradient checkpointing (when fine-tuning CLIP) and feature pre-computation (when training the Combiner network)

### Data Preparation

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

### Pre-trained models
We provide the pre-trained (both CLIP and Combiner network) checkpoint via [Google Drive](https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing) in case you don't have enough GPU resources

### CLIP fine-tuning

To fine-tune the CLIP model on FashionIQ or CIRR dataset run the following command with the desired hyper-parameters:

```sh
python src/clip_fine_tune.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --api-key {Comet-api-key} \
   --workspace {Comet-workspace} \
   --experiment-name {Comet-experiment-name} \
   --num-epochs 100 \
   --clip-model-name RN50x4 \
   --encoder both \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
```

### Combiner training

To train the Combiner model on FashionIQ or CIRR dataset run the following command with the desired hyper-parameters:

```sh
python src/combiner_train.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --api-key {Comet-api-key} \
   --workspace {Comet-workspace} \
   --experiment-name {Comet-experiment-name} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 300 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --combiner-lr 2e-5 \
   --batch-size 4096 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1
```

### Validation

To compute the metrics on the validation set run the following command

```shell
python src/validate.py 
   --dataset {'CIRR' or 'FashionIQ'} \
   --combining-function {'combiner' or 'sum'} \
   --combiner-path {path to trained Combiner} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --target-ratio 1.25 \
   --transform targetpad
```

### Test

To generate the prediction files to be submitted on CIRR evaluation server run the following command:

```shell
python src/cirr_test_submission.py 
   --submission-name {file name of the submission} \
   --combining-function {'combiner' or 'sum'} \
   --combiner-path {path to trained Combiner} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --target-ratio 1.25 \
   --transform targetpad
```


## Authors

* [**Alberto Baldrati**](https://scholar.google.it/citations?hl=en&user=I1jaZecAAAAJ)
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Tiberio Uricchio**](https://scholar.google.it/citations?user=XHZLRdYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.it/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

