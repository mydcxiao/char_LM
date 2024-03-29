# Character-Level Language Model - Baby lLama 2

This is an implementation of character-level language models based on [llama.c](https://github.com/karpathy/llama2.c) and the paper [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444).

Pretrained model weights can be found [here](https://huggingface.co/mydcxiao/char_LM).

## Comparison with other models
![comp](assets/comparison.png)

------------
## Momentum Attention

$$\text{attn} = z \cdot \text{cur-attn} + (1 - z) \cdot \text{prev-attn}$$

$$Q = \text{attn} \cdot V$$

------------

## Dataset
The training is based on [enwik8](http://mattmahoney.net/dc/enwik8.zip)

Download the dataset
```shell
python enwiki.py download
```

Tokenize and prepare DataLoader:
```shell
python enwiki.py pretokenize
```

## Train
```shell
python train.py --out_dir output --mo_attn=True
```
`--mo_attn` can set if use momentum attention or not.

Detailed configuration can be found at the beginning of [train.py](train.py)

## Test
Get the loss and bpc on val/test set of enwik8.
```shell
python test.py --checkpoint='out/ckpt.py'
```

## Generate
```shell
python generate.py --checkpoint='out/ckpt.py'
```

## Visulizaiton of Attention Map
Please run the notebook [attn_map.ipynb](attn_map.ipynb)

Comparison of last attention map with/without momentum attention
![](assets/attn_map.png)
>*left: w/o mo_attn; right: w/ mo_attn*
