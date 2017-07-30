Tree-Structured Long Short-Term Memory Networks
===============================================

An implementation of the Tree-LSTM architectures described in the paper 
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory
Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and 
Christopher Manning.

## Requirements

- [Torch7](https://github.com/torch/torch7)
- [penlight](https://github.com/stevedonovan/Penlight)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7

The Torch/Lua dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

First run the following script:

```
./fetch_and_preprocess.sh
```

This downloads the following data:

  - [SICK dataset](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) (semantic relatedness task)
  - [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) (sentiment classification task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!

and the following libraries:

  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  - [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

The preprocessing script generates dependency parses of the SICK dataset using the
[Stanford Neural Network Dependency Parser](http://nlp.stanford.edu/software/nndep.shtml).

Alternatively, the download and preprocessing scripts can be called individually.

### Semantic Relatedness

The goal of this task is to predict similarity ratings for pairs of sentences. We train and evaluate our models on the [Sentences Involving Compositional Knowledge (SICK)](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools) dataset.

To train models for the semantic relatedness prediction task on the SICK dataset,
run:

```
th relatedness/main.lua --model <dependency|constituency|lstm|bilstm> --layers <num_layers> --dim <mem_dim> --epochs <num_epochs>
```

where:

  - `model`: the LSTM variant to train (default: dependency, i.e. the Dependency Tree-LSTM)
  - `layers`: the number of layers (default: 1, ignored for Tree-LSTMs)
  - `dim`: the LSTM memory dimension (default: 150)
  - `epochs`: the number of training epochs (default: 10)

### Sentiment Classification

The goal of this task is to predict sentiment labels for sentences. For this task, we use the [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) dataset. Here, there are two sub-tasks: binary and fine-grained. In the binary sub-task, the sentences are labeled `positive` or `negative`. In the fine-grained sub-task, the sentences are labeled `very positive`, `positive`, `neutral`, `negative` or `very negative`.

To train models for the sentiment classification task on the Stanford Sentiment Treebank, run:

```
th sentiment/main.lua --model <constituency|dependency|lstm|bilstm> --layers <num_layers> --dim <mem_dim> --epochs <num_epochs>
```

This trains a Constituency Tree-LSTM model for the "fine-grained" 5-class classification sub-task.

For the binary classification sub-task, run with the `-b` or `--binary` flag, for example:

```
th sentiment/main.lua -m constituency -b
```

Predictions are written to the `predictions` directory and trained model parameters are saved to the `trained_models` directory.

See the [paper](http://arxiv.org/abs/1503.00075) for more details on these experiments.

## Third-party Implementations

- A Tensorflow Fold [re-implementation](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/sentiment.ipynb) of the Tree-LSTM for sentiment classification
