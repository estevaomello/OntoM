import time
import csv
import sys
import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

prefix_q = ''
topk = 1

def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

def run(sentences2encode, sentences2compare, fout):
    print_time()
    with open(sentences2encode) as fp:
        sentences = fp.readlines()
        sentences = [x.strip() for x in sentences] 
        print('%d sentences loaded, avg. len of %d' % (len(sentences), np.mean([len(d.split()) for d in sentences])))

    with BertClient(port=8190, port_out=5556) as bc:
        doc_vecs = bc.encode(sentences)

        with open(sentences2compare) as fq:
            sentence2comp = fq.readlines()
            sentence2comp = [x.strip() for x in sentence2comp]

        for s in sentence2comp:
            query = s
            print('%s' % query)
            query_vec = bc.encode([s])
            # compute normalized dot product as score
            score = np.sum(query_vec * doc_vecs, axis=1) / (np.linalg.norm(query_vec, axis=1) * np.linalg.norm(doc_vecs, axis=1))
            topk_idx = np.argsort(score)[::-1][:topk]
            print('top %d sentences similar to "%s"' % (topk, colored(query, 'green')))
            with open(fout, 'a+', newline='') as fa:
                fieldnames = ['score', 's1', 's2']
                writer = csv.DictWriter(fa, fieldnames=fieldnames)
                writer.writeheader()
                for idx in topk_idx:
                    writer.writerow({'score': '%.4f' % score[idx], 's1': query, 's2': sentences[idx]})
                    print('> %s\t%s' % (colored('%.4f' % score[idx], 'cyan'), colored(sentences[idx], 'yellow')))
    print_time()

if __name__== "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
