# Bringing Order to Abstractive Neural Summarization


## Overview

We present a novel training paradigm for neural abstractive summarization.
Instead of using MLE training alone, we introduce a contrastive learning component, which encourages the abstractive models to estimate the probability of system-generated summaries more accurately.

<div  align="center">
 <img src="model.png" width = "550" alt="d" align=center />
</div>



## 1. How to Install

### Requirements
- `python3.8`
- `conda create --name env --file spec-file.txt`
- Further steps
    - `pip install transformers==4.6.1`
    - `pip install pyrouge`
    - `pip install tensorboard`
    - `pip install sentencepiece`
    - `compare_mt` -> https://github.com/neulab/compare-mt
        ```console
        git clone https://github.com/neulab/compare-mt.git
        cd ./compare-mt
        pip install -r requirements.txt
        python setup.py install
        ```
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

### Preprocessed Data

You can download the preprocessed data for our experiments on [CNNDM](https://drive.google.com/file/d/19TLcmoKnssPLQT4LJS_B7nm8KBT9WuyZ/view?usp=sharing).

After donwloading, you should unzip the zip files in this root directory.

For NYT, you will need to get the license and please follow https://github.com/kedz/summarization-datasets for pre-processing.

### Preprocess Your Own Data

For data preprocessing, please run
```console
python preprocess.py --src_dir [path of the raw data] --tgt_dir [output path] --split [train/val/test] --cand_num [number of candidate summaries] --dataset [cnndm/xsum/nyt] -l [lowercase if the flag is set]
```
`src_dir` should contain the following files (using test split as an example):
- `test.source`
- `test.source.tokenized`
- `test.target`
- `test.target.tokenized`
- `test.out`
- `test.out.tokenized`

Each line of these files should contain a sample except for `test.out` and `test.out.tokenized`. In particular, you should put the candidate summaries for one data sample at neighboring lines in `test.out` and `test.out.tokenized`.

We use the PTB tokenizer provided by Standford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) ([download here](https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.8.0/stanford-corenlp-3.8.0.jar)). Please not that tokenized texts are *only* used for evaluation.

We have provided the examples files in `./examples/raw_data`.

The preprocessing precedure will store the processed data as seperate json files in `tgt_dir`.

**Example: preprocessing test set on CNNDM**

```console
# starting from the root directory

# create folders
mkdir ./cnndm
mkdir ./cnndm/diverse
mkdir ./cnndm/diverse/test

# suppose that the raw files at ./raw_data 

python preprocess.py --src_dir ./raw_data --tgt_dir ./cnndm/diverse --split test --cand_num 16 --dataset cnndm -l
```


## 3. How to Run


### Hyper-parameter Setting
You may specify the hyper-parameters in `main.py`.
We also provide the specific settings on CNNDM (NYT share the same setting) and XSum in `config.py`.

### Train
```
python main.py --cuda --gpuid [list of gpuid] --config [name of the config (cnndm/xsum)] -l 
```
**Example: training on CNNDM**
```
python main.py --cuda --gpuid 0 1 2 3 --config cnndm -l 
```

**Finetuning from an existing checkpoint**
```
python main.py --cuda --gpuid [list of gpuid] -l --config [name of the config (cnndm/xsum)] --model_pt [model path]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

### Evaluate
For ROUGE calculation, we use the standard ROUGE Perl package from [here](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5). Note that the scores calculated by this package would be sightly *different* from the ROUGE scores calculated/reported during training/intermidiate stage of evalution, because we use a pure python-based ROUGE implemenatation to calculate those scores for better efficiency. 

We lowercased and tokenized (PTB Tokenizer) texts before calculating the ROUGE scores.

```
python main.py --cuda --gpuid [single gpu] --config [name of the config (cnndm/xsum)] -e --model_pt [model path] -g [evaluate the model as a generator] -r [evaluate the model as a scorer/reranker]
```
model path should be a subdirectory in the `./cache` directory, e.g. `cnndm/model.pt` (it shouldn't contain the prefix `./cache/`).

**Example: evaluating the model as a generator on CNNDM**
```
python main.py --cuda --gpuid 0 --config cnndm -e --model_pt cnndm/model_generation.bin -g
```

**Example: evaluating the model as a scorer on CNNDM**
```
python main.py --cuda --gpuid 0 --config cnndm -e --model_pt cnndm/model_ranking.bin -r
```

## 4. Results

### CNNDM
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 44.16   | 21.28   | 40.90   |
| Ours     | 47.78   | 23.55   | 44.57   |

### XSum
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| Pegasus  | 47.21   | 24.56   | 39.25   |
| Ours     | 49.07   | 25.59   | 40.40   |

### NYT
|          | ROUGE-1 | ROUGE-2 | ROUGE-L |
|----------|---------|---------|---------|
| BART     | 55.78   | 36.61   | 52.60   |
| Ours     | 57.75   | 38.64   | 54.54   |

Our model outputs on these datasets can be found in `./output`.

We have also provided the finetuned checkpoints on [CNNDM](https://drive.google.com/drive/folders/1mdfYcHF9OfVb0eAggzaIfNk-63hnCeqh?usp=sharing), [XSum](https://drive.google.com/drive/folders/1EnHRuzH0rVIKLrseN8xqgvIgpakgrCxJ?usp=sharing) and [NYT](https://drive.google.com/drive/folders/1WriaJ2ozVlof0zNHjeqsDxcfpavL4ApB?usp=sharing).

You could load these checkpoints using the standard Transformers' interface (model.from_pretrained()).