'''

'''
import os
import numpy as np
import json
from collections import Counter

class Utils:

    # iterate json
    @staticmethod
    def scan_json(indir:str):
        for root, dirs, files in os.walk(indir):
            for file in files:
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    data = json.load(f)
                    yield (data, path)

    @staticmethod
    def expand(seq_len:int, start:int, end:int, size:int):
        '''
        given a sequence with index 0-<seq_len>
        symetrically expand window <start, end>to the <size>
        arg: <start> and <end> are index
        Note: seq_len > end > start >= 0, seq_len > size, end-start<size
        '''
        if end-start >= size or size > seq_len or \
            start < 0 or end < 0 or start > end:
            return None
        pos = 'end'
        while (end - start) < size:
            if pos == 'end':
                if end + 1 < seq_len:
                    end += 1
                pos = 'start'
            elif pos == 'start':
                if start - 1 >= 0:
                    start -= 1
                pos = 'end'
        return start, end

    @staticmethod
    def shrink(start:int, end:int, size:int):
        '''
        symetrically shrink window <start, end> to the <size>
        arg: <start> and <end> are index
        Note: end > start >= 0, size > end-start > 0
        '''
        if end-start <= size or start < 0 or end < 0 \
            or start > end or size < 1:
            return None
        pos = 'end'
        while (end - start) > size:
            if pos == 'end':
                if end - 1 > start:
                    end -= 1
                pos = 'start'
            elif pos == 'start':
                if start + 1 < end:
                    start += 1
                pos = 'end'
        return start, end