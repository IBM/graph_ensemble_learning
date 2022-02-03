import warnings
import argparse

import penman
import torch
from amrlib.utils.logging import silence_penman
from penman.models.noop import NoOpModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from amr_utils.datasets.dataset import AMRPenman,  add_prefix, AMR_GENERATION
from amr_parsing.t5.cli.inference import Inference
from amr_parsing.t5.models.lg import LG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.simplefilter('ignore')


def parse(batch_size, root, num_beams, num_ret_seq, output_file):
    silence_penman()
    data_set = AMRPenman(root)

    valid_set = DataLoader(
        data_set, batch_size=2*batch_size, num_workers=4, shuffle=False
    )
    dataset = iter(valid_set)
    net.eval()
    inference = Inference(net)
    amrs = []
    amr_predictions = []

    with torch.no_grad():
        for amr_seq, sentence, amr in tqdm(dataset):
            gids = []
            for a in amr:
                g = penman.decode(a, model=NoOpModel())
                if 'id' in g.metadata:
                    gid = g.metadata['id']
                else:
                    gid = None
                gids.append(gid)
            predict_amrs = inference.parse_sentences(add_prefix(sentence, AMR_GENERATION), num_beams, num_ret_seq)
            amrs += amr
            ps = []
            for i in range(len(predict_amrs)):
                p = predict_amrs[i]
                gid = gids[i]
                s = sentence[i]
                if p is not None:
                    ps.append('# ::snt ' + s + '\n' + '# ::id ' + gid + '\n' + p)
                else:
                    ps.append(None)
            amr_predictions += ps

    num_none = len([p for p in amr_predictions if p is None])
    print("Number of None:", num_none)
    f_gen = open(output_file, 'wt')
    print('Saving %s' % f_gen)
    skipped = 0
    for ref_graph, gen_graph in zip(amrs, amr_predictions):
        if gen_graph is None:
            skipped += 1
            continue
        f_gen.write(gen_graph + '\n\n')
    f_gen.close()
    print('Out of %d graphs, skipped %d that did not deserialize properly.' % (len(amrs), skipped))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMR parse')
    parser.add_argument(
        '-t', '--test', default='./test.datasets', type=str,
        help='Root directory with the test datasets')

    parser.add_argument(
        '-b', '--batch', type=int, default=4, help='Mini-batch size')
    parser.add_argument(
        '-o', '--output', type=str, default='./pred.text',
        help='Output folder where the model, and output files will be pickled')
    parser.add_argument(
        '-m', '--model', type=str, default='t5-base', help='model name')
    parser.add_argument(
        '--max_source_length', type=int, default=16, help='Max source length')
    parser.add_argument(
        '--max_target_length', type=int, default=16, help='Max target length')
    parser.add_argument(
        '--num_beams', type=int, default=5, help='Number of beams used during inference')
    parser.add_argument(
        '--num_ret_seq', type=int, default=5, help='Number of return sequences for each prediction during inference')
    parser.add_argument(
        '--model_type', type=str, default='t5', help='Model type: bart or t5')
    parser.add_argument(
        '-c', '--checkpoint', type=str, default=None, help='Checkpoint model')

    parser.add_argument('--data_type', default="blinkify", help='Type of datasets')
    parser.add_argument('--task_type', default="text2amr", help='Type of datasets')

    args = parser.parse_args()

    net = LG(args.model,
             max_source_length=args.max_source_length,
             max_target_length=args.max_target_length,
             model_type=args.model_type)

    if args.checkpoint is not None:
        print("Load model from ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        if torch.cuda.device_count() <= 1:
            net.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.model.module.load_state_dict(checkpoint['model_state_dict'])

    parse(args.batch,
          args.test,
          args.num_beams,
          args.num_ret_seq,
          args.output)
