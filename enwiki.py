"""
Download, preprocess and serve the enwiki dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import sys
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
# import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm
import zipfile

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
VOCAB_SIZE = 0

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the enwiki dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the enwiki dataset, unless it's already downloaded
    data_url = "http://mattmahoney.net/dc/enwik8.zip"
    data_filename = os.path.join(DATA_CACHE_DIR, "enwik8.zip")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")
    print("Download done.")

def pretokenize():
    """Unpacks the enwiki dataset to DATA_CACHE_DIR"""
    if os.path.exists(os.path.join(DATA_CACHE_DIR, 'train_tok.bin')):
        print('Tokenized enwik8 already exists - skipping processing')
        # sys.exit()
        return
    
    data_filename = os.path.join(DATA_CACHE_DIR, "enwik8.zip")
    data = zipfile.ZipFile(data_filename).read('enwik8')
    print('Length of enwik8: {}'.format(len(data)))

    num_test_chars = 5000000

    train_data = data[: -2 * num_test_chars]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    test_data = data[-num_test_chars:]
    
    train_filename = "train"
    valid_filename = "valid"
    test_filename = "test"
    
    enc = Tokenizer()
    global VOCAB_SIZE
    for fn, part in [(train_filename, train_data), (valid_filename, valid_data), (test_filename, test_data)]:
        print('{} will have {} bytes'.format(fn, len(part)))
        print('- Tokenizing ...')
        all_tokens = []
        s = []
        for i in range(len(part)):
            c = part[i]
            if c == ord('\n'):
                s = "".join([chr(c) for c in s])
                tokens = enc.encode(s, bos=True, eos=False)
                all_tokens.extend(tokens)
                s = []
            else:
                s.append(c)
            VOCAB_SIZE = max(VOCAB_SIZE, c + enc.n_special_tokens)
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        tokenized_filename = os.path.join(DATA_CACHE_DIR, f"{fn}_tok.bin")
        # write the bytes
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")
        print('Vocab size:', VOCAB_SIZE)
    print('Tokenizing done.')
        

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_source, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "enwik8":
            # the .bin files are right along the .json files
            bin_dir = DATA_CACHE_DIR
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = DATA_CACHE_DIR
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = [shard_filename for shard_filename in shard_filenames if self.split in shard_filename]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the tokenizer:
    python enwiki.py download
    python enwiki.py pretokenize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize"])
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")
    
    # iter_batches = partial(
    # Task.iter_batches,
    # batch_size=2,
    # max_seq_len=500,
    # # vocab_size=vocab_size,
    # vocab_source="enwik8",
    # device="cuda",
    # num_workers=0,
    # )
    
    # batch_iter = iter_batches(split="train")
    
    # print(next(batch_iter))