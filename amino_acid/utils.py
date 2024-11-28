'''

'''
import os
import numpy as np
import pandas as pd
import json
from collections import Counter

class Utils:

    @staticmethod
    def scan_text(indir:str, sep):
        for root, dirs, files in os.walk(indir):
            for file_name in files:
                path = os.path.join(root, file_name)
                df = pd.read_csv(path, sep=sep, header=None, index_col=None)
                yield df, file_name

    # iterate json
    @staticmethod
    def scan_json(indir:str):
        for root, dirs, files in os.walk(indir):
            for file_name in files:
                path = os.path.join(root, file_name)
                with open(path, 'r') as f:
                    data = json.load(f)
                    yield data, file_name

    @staticmethod
    def scan_json_record(indir:str):
        data_iter = Utils.scan_json(indir)
        for data, _ in data_iter:
            for acc in data:
                yield acc, data[acc]

    @staticmethod
    def scan_json_seq(indir:str, key:str):
        data_iter = Utils.scan_json(indir)
        for data, _ in data_iter:
            for acc in data:
                items = data[acc][key]
                for item in items.values():
                    yield item['seq']

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
        n = 0
        while (end - start) < size:
            n += 1
            if pos == 'end':
                if end + 1 < seq_len:
                    end += 1
                pos = 'start'
            elif pos == 'start':
                if start - 1 >= 0:
                    start -= 1
                pos = 'end'
            if n > size * 3:
                return None
        return start, end

    @staticmethod
    def shrink(start:int, end:int, size:int):
        '''
        symetrically shrink window <start, end> to the <size>
        arg: <start> and <end> are index
        Note: end > start >= 0, size > end-start > 0
        '''
        if end-start <= size or start < 0 or end < 0 \
            or start > end or size <= 1:
            return None
        pos = 'end'
        n = 0
        while (end - start) > size:
            n += 1
            if pos == 'end':
                if end - 1 > start:
                    end -= 1
                pos = 'start'
            elif pos == 'start':
                if start + 1 < end:
                    start += 1
                pos = 'end'
            if n > size * 3:
                return None
        return start, end