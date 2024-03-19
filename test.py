import math
import os
import time
import glob
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import numpy as np

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from enwiki import Task


# -----------------------------------------------------------------------------
checkpoint = 'out/ckpt.pt'
# eval_iters = 100
data_cache_dir = "data"
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 512
vocab_source = "enwik8" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 244 # the Llama 2 tokenizer has 32K tokens
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)

iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

@torch.no_grad()
def estimate_loss_bpc():
    out = {}
    model.eval()
    for split in ["train", "val", "test"]:
        batch_iter = iter_batches(split=split)
        if vocab_source == "enwik8":
            # the .bin files are right along the .json files
            bin_dir = data_cache_dir
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = data_cache_dir
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = [shard_filename for shard_filename in shard_filenames if split in shard_filename]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        all_batches = 0
        for shard in shard_filenames:
            # open the dataset for reading but keep it on disk with memmap
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            num_batches = len(m) // max_seq_len
            num_batches -= 1
            all_batches += num_batches
        eval_iters = all_batches // batch_size
        losses = torch.zeros(eval_iters)  # keep on CPU
        bpcs = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss
                bpc = model.last_bpc
            losses[k] = loss.item()
            bpcs[k] = bpc.item()
        out[split] = (losses.mean(), bpcs.mean())
    return out

loss_bpc = estimate_loss_bpc()
print(f"Train loss: {loss_bpc['train'][0]:.4f}, Val loss: {loss_bpc['val'][0]:.4f}, Test loss: {loss_bpc['test'][0]:.4f}")
print(f"Train bpc: {loss_bpc['train'][1]:.4f}, Val bpc: {loss_bpc['val'][1]:.4f}, Test bpc: {loss_bpc['test'][1]:.4f}")