# Bringing Order to Abstractive Neural Summarization


## Overview

We present a novel training paradigm for neural abstractive summarization.
Instread of using MLE training alone, we introduce a contrastive learning component, which encourages the abstractive models to estimate the probability of system-generated summaries more accurately.

<div  align="center">
 <img src="model.png" width = "550" alt="d" align=center />
</div>



## 1. How to Install

### Requirements
- `python3`
- `conda create --name env --file spec-file.txt`
- `pip3 install -r requirements.txt`
- `compare_mt` -> https://github.com/neulab/compare-mt
Our code is based on Huggingface's [Transformers](https://github.com/huggingface/transformers) library. 

### Description of Codes
- `main.py` -> training and evaluation procedure
- `model.py` -> models
- `modeling_bart.py`, `modeling_pegasus.py` -> modefied from Transformers library to support more efficient training
- `label_smoothing_loss.py` -> label smoothing loss
- `data_utils.py` -> dataloader
- `utils.py` -> utility functions
- `preprocess.py` -> data preprocessing

### Workspace
Following directories should be created for our experiments.
- `./cache` -> storing model checkpoints
- `./result` -> storing evaluation results

## 2. Preprocessing

We use the following datasets for our experiments.

- CNN/DailyMail -> https://github.com/abisee/cnn-dailymail
- XSum -> https://github.com/EdinburghNLP/XSum
- NYT -> https://catalog.ldc.upenn.edu/LDC2008T19

For data preprocessing, please run
```
python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries]
```
`src_dir` should contain the following files (using test split as an example):
- `test.source`
- `test.source.tokenized`
- `test.target`
- `test.target.tokenized`
- `test.out`
- `test.out.tokenized`

Each line of these files should contain a sample. In particular, you should put the candidate summaries for one data sample at neighboring lines in `test.out` and `test.out.tokenized`.

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.


## 3. How to Run

### Preprocessed Data
You can download the preprocessed data for our experiments on [CNNDM](https://drive.google.com/file/d/1WRvDBWfmC5W_32wNRrNa6lEP75Vx5cut/view?usp=sharing) and [XSum](https://drive.google.com/file/d/1nKx6RT4zNxO4hFy8y3dPbYV-GBu1Si-u/view?usp=sharing).

After donwloading, you should unzip the zip files in this root directory.

For NYT, you will need to get the license and please follow https://github.com/kedz/summarization-datasets for pre-processing.

### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.

### Train
```
python main.py --cuda --gpuid [list of gpuid] -l
```
### Fine-tune
```
python main.py --cuda --gpuid [list of gpuid] -l --model_pt [model path]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).
### Evaluate
```
python main.py --cuda --gpuid [single gpu] -e --model_pt [model path]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

## 4. Results

### CNNDM
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 44.39   | 21.21   | 41.28   |
| Ours     | 46.67   | 22.15   | 43.54   |

### XSum
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Pegasus  | 47.10   | 24.53   | 39.23   |
| Ours     | 47.61   | 24.57   | 39.44   |

Our model outputs on these datasets can be found in `./output`.

We have also provided the finetuned checkpoints on [CNNDM](https://drive.google.com/file/d/1CSFeZUUVFF4ComY6LgYwBpQJtqMgGllI/view?usp=sharing) and [XSum](https://drive.google.com/file/d/1yx9KhDY0CY8bLdYnQ9XhvfMwxoJ4Fz6N/view?usp=sharing).