import argparse
import logging
import os

from amr_utils.blinkify.blink_wiki_adder import BlinkWikiAdder

logger = logging.getLogger(__name__)

# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blink Wiki adder')
    parser.add_argument('--blink-models-dir', type=str, required=True)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('-i', '--input', required=True, type=str, help='Amr file without the Wiki')
    parser.add_argument('-o', '--output', required=True, type=str, help='Root directory for output')
    args = parser.parse_args()

    input_file = args.input
    input_file_name = os.path.basename(input_file)
    out_dir = args.output

    os.makedirs(out_dir, exist_ok=True)

    # run the pipeline
    out_fn = os.path.join(out_dir, input_file_name + '.wiki')
    wiki = BlinkWikiAdder(models_path=args.blink_models_dir, fast=args.fast)
    print('Wikifing', input_file)
    wiki.wikify_file(input_file, out_fn)
    print('Data written to', out_fn)
