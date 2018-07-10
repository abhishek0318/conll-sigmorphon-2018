import os
import subprocess

from constants import *
from data import read_dataset


def one_one_alignment(srcs, tgts):
    """Gets alignments on how seq2 is aligned to seq1.

    Args:
        srcs: list of sequences, eg. [['i', 'n', 'd', 'i', 'a'], .. ]
        tgts: list of sequences eg. [['I', 'N', 'D', 'I', 'A], .. ]

    Returns:
        alignment: list of list containing index which ith token of tgt is mapped to
                    eg. [[0, 1, 2, 3, 4], .. ]
    """

    with open(os.path.join(M2M_ALIGNER_PATH, 'input'), 'w', encoding='utf8') as file:
        for src, tgt in zip(srcs, tgts):
            src = [token.replace(' ', '*') for token in src]
            tgt = [token.replace(' ', '*') for token in tgt]
            file.write(' '.join(src) + '\t' + ' '.join(tgt) + '\n')

    subprocess.run([os.path.join(M2M_ALIGNER_PATH, 'm2m-aligner'), '--errorInFile', '--delY',  '--maxX', '1',
                    '--maxY', '1', '-i', os.path.join(M2M_ALIGNER_PATH, 'input'), '-o', os.path.join(M2M_ALIGNER_PATH, 'output')])

    alignments = []

    with open(os.path.join(M2M_ALIGNER_PATH, 'output'), 'r', encoding='utf8') as file:
        for i, line in enumerate(file):
            if 'NO ALIGNMENT' in line:
                alignments.append([-1] * len(tgts[i]))
                continue

            src, tgt = line.split('\t')
            src = src.replace('|', '')
            tgt = tgt.replace('|', '')

            seq_alignment = []
            src_i = 0
            for token1, token2 in zip(src, tgt):
                if token1 != '_':
                    seq_alignment.append(src_i)
                    src_i +=1
                else:
                    seq_alignment.append(-1)

            alignments.append(seq_alignment)

    return alignments


if __name__ == "__main__":
    lemmas, tags, inflected_forms = read_dataset(os.path.join(TASK1_DATA_PATH, 'hindi-train-high'))
    alignments = one_one_alignment([list(word) for word in lemmas], [list(word) for word in inflected_forms])
    print(alignments)