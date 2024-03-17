import os
import struct
import argparse
from typing import List

TOKENIZER_MODEL = "tokenizer.model"

class Tokenizer:
    def __init__(self):      
        # BOS / EOS token IDs
        self.n_special_tokens: int = 3 
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.pad_id: int = 0
        #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = []
        for c in s:
            t.append(ord(c) + self.n_special_tokens)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        s = []
        for i in t:
            if i == self.bos_id:
                s.append("\n")
            elif i == self.eos_id:
                break
            elif i == self.pad_id:
                break
            else:
                s.append(chr(i - self.n_special_tokens))
        return ''.join(s)
            