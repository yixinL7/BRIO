from os.path import join
import logging
import tempfile
import subprocess as sp

from pyrouge import Rouge155
from pyrouge.utils import log
import os
import argparse
from nltk import sent_tokenize
from compare_mt.rouge.rouge_scorer import RougeScorer

_ROUGE_PATH = '/YOUR-ABSOLUTE-PATH/ROUGE-RELEASE-1.5.5/'

def eval_rouge(dec_dir, ref_dir, Print=False):
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = ('perl ' + join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
            + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
            + cmd
            + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        print(cmd)
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
        R_1 = float(output.split('\n')[3].split(' ')[3])
        R_2 = float(output.split('\n')[7].split(' ')[3])
        R_L = float(output.split('\n')[11].split(' ')[3])
        print(output)
    if Print is True:
        rouge_path = join(dec_dir, '../ROUGE.txt')
        with open(rouge_path, 'w') as f:
            print(output, file=f)
    return R_1, R_2, R_L

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--ref", type=str, help="path of a directory or a file containing reference summaries", required=True)
    parser.add_argument("--hyp", type=str, help="path of a directory or a file containing candidate summaries", required=True)
    parser.add_argument("-p", "--python", action="store_true", help="use python rouge")
    parser.add_argument("-l", "--lower", action="store_true", help="lowercase")
    args = parser.parse_args()
    if not os.path.isdir(args.ref):
        # if args.ref is a file, generate a directory to store the summaries
        ref_dir = tempfile.mkdtemp()
        with open(args.ref, 'r') as f:
            for (i, line) in enumerate(f):
                line = line.strip()
                if args.lower:
                    line = line.lower()
                with open(join(ref_dir, f"{i}.ref"), 'w') as f2:
                    for x in sent_tokenize(line):
                        print(x, file=f2)
        hyp_dir = tempfile.mkdtemp()
        with open(args.hyp, 'r') as f:
            for (i, line) in enumerate(f):
                line = line.strip()
                if args.lower:
                    line = line.lower()
                with open(join(hyp_dir, f"{i}.dec"), 'w') as f2:
                    for x in sent_tokenize(line):
                        print(x, file=f2)
    else:
        ref_dir = args.ref
        hyp_dir = args.hyp
    if args.python:
        rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0
        num = len(os.listdir(ref_dir))
        for i in range(num):
            ref = open(join(ref_dir, f"{i}.ref"), 'r').read()
            hyp = open(join(hyp_dir, f"{i}.dec"), 'r').read()
            scores = rouge_scorer.score(ref, hyp)
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeLsum += scores['rougeLsum'].fmeasure
            cnt += 1
        print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1 / cnt * 100, rouge2 / cnt * 100, rougeLsum / cnt * 100))
    else:
        eval_rouge(hyp_dir, ref_dir, Print=True)
