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
