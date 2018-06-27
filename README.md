## TypeSQL

Source code accompanying our NAACL 2018 paper:[TypeSQL: Knowledge-based Type-Aware Neural Text-to-SQL Generation
](https://arxiv.org/abs/1804.09769)

#### Environment Setup

1. The code uses Python 2.7 and [Pytorch 0.2.0](https://pytorch.org/previous-versions/) GPU.
2. Install Python dependency: `pip install -r requirements.txt`

#### Download Data and Embeddings

1. Download the zip data file at the [Google Drive](), and put it in the root dir.
2. Download the pretrained [Glove](https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) and the [paraphrase embedding](https://drive.google.com/file/d/1rbF3daJjCsa1-fu2GANeJd2FBXos1ugD/view) `para-nmt-50m/data/paragram_sl999_czeng.txt`

#### Train a Model

1. To use knowledge graph types:
```
  `mkdir saved_model_kg`
  `python train.py --sd saved_model_kg`
```

2. To use DB content types:
```
   `mkdir saved_model_con`
   `python train.py --sd saved_model_con --db_content 1`
```

#### Test a Model

1. `python test.py --sd saved_model_kg` or `python test.py --sd saved_model_con --db_content 1`
