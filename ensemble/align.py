import argparse

import penman
from amrlib.evaluate.smatch_enhanced import compute_smatch
from ensemble.utils import align, get_entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Ensemble (Graphene)')
    parser.add_argument(
        '-g', '--gold', default='./datasets/spring_gold_bio.txt',
        type=str,
        help='Gold amr file')
    parser.add_argument(
        '-p', '--prediction', default='./datasets/graphene_bio_all.wiki.txt',
        type=str,
        help='Prediction files')

    args = parser.parse_args()
    ref_fname = args.gold
    print('Gold file:', ref_fname)
    gen_fname = args.prediction
    original_gold_entries, gold_entries = get_entries(ref_fname)
    print('Prediction file:', gen_fname)
    original_test_entries_1, test_entries_1 = get_entries(gen_fname)
    print("Align files")
    test = align(original_gold_entries, original_test_entries_1, test_entries_1)
    precision, recall, f_score = compute_smatch(test, gold_entries)
    print(' SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))

    test = [penman.encode(penman.decode(g)) for g in test]
    outputs = []
    for g, p in zip(original_gold_entries, test):
        r = penman.decode(g)
        s = '# ::snt ' + r.metadata['snt'] + '\n' + '# ::id ' + r.metadata['id'] + '\n' + p
        outputs.append(s)

    output_file = args.prediction + '.aligned'
    with open(output_file, 'wt') as f:
        print('Write prediction to', output_file)
        f.write('\n\n'.join(map(str, outputs)))





