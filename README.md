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

**For the semantic relatedness task, run:**

```
th relatedness/main.lua
```

**For the sentiment classification task, run:**

```
th sentiment/main.lua
```

This trains a model for the "fine-grained" 5-class classification sub-task.

For the binary classification sub-task, run:

```
th sentiment/main.lua --binary
```

Predictions are written to the `predictions` directory and trained model parameters are saved to the `trained_models` directory.

See the [paper](http://arxiv.org/abs/1503.00075) for details on these experiments.
