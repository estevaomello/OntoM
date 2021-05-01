import time
import pandas as pd
import csv
import sys
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

def run(sentences2encode, fout):
    print_time()
    with open(sentences2encode) as fp:
        sentences = fp.readlines()
        sentences = [x.strip() for x in sentences] 
        print('%d sentences loaded, avg. len of %d' % (len(sentences), np.mean([len(d.split()) for d in sentences])))

    with BertClient(port=8190, port_out=5556) as bc:
        res = bc.encode(sentences)
        df = pd.DataFrame(data=res.astype(float))
        df.to_csv(fout, sep=',', header=False, float_format='%.4f', index=False)

    print_time()

if __name__== "__main__":
    run(sys.argv[1], sys.argv[2])
