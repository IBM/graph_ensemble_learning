import os
import logging
from amrlib.utils.logging import silence_penman
from amrlib.graph_processing.wiki_remover import wiki_remove_file
from amrlib.graph_processing.amr_loading_raw import load_raw_amr
import argparse
from .utils import annotate_file, load_spacy
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    """
    - Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
      plus a few other fields for future pre/postprocessing work that may be needed.
    - Remove Wiki tag from the AMRs
    """
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument(
        '-i', '--input', default='./LDC2017T10/datasets/amrs/split', type=str,
        help='Root directory with the datasets')
    parser.add_argument(
        '-o', '--output', default='./LDC2017T10/preprocessed_data/', type=str,
        help='Root directory for output')

    parser.add_argument(
        '-s', '--spacy_model', default='en_core_web_sm', type=str,
        help='Spacy model')

    args = parser.parse_args()

    base_dir = args.input
    out_dir = args.output

    os.makedirs(out_dir, exist_ok=True)

    # Loop through the directories
    for dirname in ('dev', 'test', 'training'):
        entries = []
        dn = os.path.join(base_dir, dirname)
        print('Loading datasets from', dn)
        fpaths = [os.path.join(dn, fn) for fn in os.listdir(dn)]
        for fpath in fpaths:
            entries += load_raw_amr(fpath)
        print('Loaded {:,} entries'.format(len(entries)))
        # Save the collated datasets
        fn = 'train.txt' if dirname == 'training' else dirname + '.txt'
        out_path = os.path.join(out_dir, fn)
        print('Saving datasets to', out_path)
        with open(out_path, 'w') as f:
            for entry in entries:
                f.write('%s\n\n' % entry)
        print()

    silence_penman()

    # Create the processed corpus directory
    os.makedirs(out_dir, exist_ok=True)

    # Load the spacy model with the desired models
    spacy_nlp = load_spacy(args.spacy_model)

    # run the pipeline
    for fn in ('test.txt', 'dev.txt', 'train.txt'):
        print("Annotating ", fn)
        annotate_file(out_dir, fn, out_dir, fn + '.features', spacy_nlp)

    for fn in ('test.txt.features', 'dev.txt.features', 'train.txt.features'):
        print("Remove wiki ", fn)
        wiki_remove_file(out_dir, fn, out_dir, fn + '.nowiki')

