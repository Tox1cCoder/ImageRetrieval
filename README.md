# Content-Based Image Retrieval

## Introduction

This project tackles Content-based Image Retrieval using the Faiss algorithm by Facebook. It integrates multiple feature extraction methods like SIFT, Local Binary Pattern, and ResNet50 for comparison and evaluation purposes.

**Problem Statement**

  - **Input**: Image database, query image
  - **Output**: Ranked list of images similar to the query, with the most similar on top

Utilizing the [faiss](https://github.com/facebookresearch/faiss.git) library by Facebook, ResNet50 network weights are sourced from the pre-trained model in [torchvision.models](https://pytorch.org/vision/stable/models.html).

## Prepare the environment

1. Install PyTorch-cuda==11.7 following [official instruction](https://pytorch.org/):

        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
        
2. Install FAISS:

        conda install -c conda-forge faiss-gpu
        
3. Install the necessary dependencies by running:

        pip install -r requirements.txt

## Prepare the dataset

1. Download [The Paris Dataset](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) (or via [Kaggle](https://www.kaggle.com/datasets/skylord/oxbuildings?select=paris_2.tgz)) and place it in ./data/paris.

2. Obtain the corresponding [groundtruth](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) and place it in ./data/groundtruth.

Organize your dataset following this structure:

```
Main-folder/
│
├── dataset/ 
│   ├── evaluation
|   |   ├── crop
|   |   |   ├── LBP
|   |   |   |   ├── defense_1.txt
|   |   |   |   ├── eiffel_1.txt
|   |   |   |   └── ...
|   |   |   ├── Resnet50
|   |   |   |   └── ...
|   |   |   └── SIFT
|   |   |       └── ...
|   |   └── original
|   |       └── ...
|   |
│   ├── feature
|   |   ├── LBP.index.bin
|   |   ├── Resnet50.index.bin
|   |   └── SIFT.index.bin
|   |   
|   ├── groundtruth
|   |   ├── defense_1_good.txt
|   |   ├── louvre_2_junk.txt
|   |   └── ...
|   |
|   └── paris
|       ├── paris_defense_000000.jpg
|       ├── paris_moulinrouge_000164.jpg
|       └── ...
|   
└── ...
```

## Running the code

### Feature extraction (Indexing)

    python indexing.py --feature_extractor Resnet50
    
The Resnet50.index.bin file will be at **Main-folder/dataset/feature**.

### Evaluation

Evaluation on query set

    python ranking.py --feature_extractor Resnet50
    
### Compute Mean Average Precision (MAP):

    python evaluate.py --feature_extractor Resnet50
    
### Run demo with streamlit interface:

    streamlit run demo.py
    
### Configuration 

You can modify the config like feature_extractor (SIFT, LBP, Resnet50), batch_size, top_k, ...