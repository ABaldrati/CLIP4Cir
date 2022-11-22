# CLIP4Cir

### CLIP for Composed image retrieval

## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Composed image retrieval task

![](images/cir-overview.png "Composed image retrieval overview")

### CLIP task-oriented fine-tuning 

![](images/clip-fine-tuning.png "CLIP task oriented fine-tuning")

### Combiner training 

![](images/combiner-training.png "Combiner training")

### Combiner architecture

![](images/Combiner-architecture.png "Combiner architecture overview")

### Abstract

Recent works have shown that large-scale vision and language pretrained (VLP) models can address many different tasks, such as zero-shot learning or text-to-image retrieval.
In this paper, we address the task of composed image retrieval. In this recently introduced task, the query is provided as an image-text pair. Multi-modal content-based image retrieval is performed starting with a reference image and an additional text that describes in natural language conditions or changes with respect to the reference image, about the output images to be retrieved.

To address this task, we explore the use of features obtained from the OpenAI CLIP model, and we initially perform a task-oriented fine-tuning of both CLIP encoders using a combination of visual and textual features. Then, in the second stage, we learn a Combiner network that can merge the fine-tuned features integrating the multimodal information and providing combined features used to perform the retrieval task. Contrastive learning is used in the training of both stages.

Starting from the bare CLIP features as a baseline, we show that both the task-oriented fine-tuning and the carefully crafted Combiner network are highly effective and outperform more complex state-of-the-art approaches on FashionIQ and CIRR, two popular and challenging datasets for composed image retrieval

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [CLIP](https://github.com/openai/CLIP)
* [Comet](https://www.comet.ml/site/)

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

```shell
python src/clip_fine_tune.py --dataset {'CIRR' or 'FashionIQ'} --api-key {Comet-api-key} --workspace {Comet-workspace} --experiment-name {Comet-experiment-name} --num-epochs 100 --clip-model-name RN50x4 --encoder both --learning-rate 2e-6 --batch-size 128 --transform targetpad --target-ratio 1.25  --save-training --save-best --validation-frequency 1
```

### Combiner training

To train the Combiner model on FashionIQ or CIRR dataset run the following command with the desired hyper-parameters:

```shell
python src/combiner_train.py --dataset {'CIRR' or 'FashionIQ'} --api-key {Comet-api-key} --workspace {Comet-workspace} --experiment-name {Comet-experiment-name} --projection-dim 2560 --hidden-dim 5120 --num-epochs 300 --clip-model-name RN50x4 --clip-model-path {path-to-fine-tuned-CLIP} --combiner-lr 2e-5 --batch-size 4096 --clip-bs 32 --transform targetpad --target-ratio 1.25 --save-training --save-best --validation-frequency 1
```

### Validation

To compute the metrics on the validation set run the following command

```shell
python src/validate.py --dataset {'CIRR' or 'FashionIQ'} --combining-function {'combiner' or 'sum'} --combiner-path {path to trained Combiner} --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --clip-model-path {path-to-fine-tuned-CLIP} --target-ratio 1.25 --transform targetpad
```

### Test

To generate the prediction files to be submitted on CIRR evaluation server run the following command:

```shell
python src/cirr_test_submission.py --submission-name {file name of the submission} --combining-function {'combiner' or 'sum'} --combiner-path {path to trained Combiner} --projection-dim 4096 --hidden-dim 8192 --clip-model-name RN50x4 --clip-model-path {path-to-fine-tuned-CLIP} --target-ratio 1.25 --transform targetpad
```

## Authors

* [**Alberto Baldrati**](https://scholar.google.it/citations?hl=en&user=I1jaZecAAAAJ)
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Tiberio Uricchio**](https://scholar.google.it/citations?user=XHZLRdYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.it/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research. This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

## License

## Citation

## Contacts
