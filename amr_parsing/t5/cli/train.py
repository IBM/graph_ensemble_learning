import argparse
import os
import random
import warnings
from pathlib import Path

import penman
import torch
from amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from amrlib.utils.logging import silence_penman
from penman.models.noop import NoOpModel
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from amr_utils.datasets.dataset import AMRPenman, AMRPenmanMultiTasks, add_prefix, CombineData, AMR_GENERATION
from .inference import Inference
from ..models.lg import LG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.simplefilter('ignore')


def save(optimizer, model, path):
    Path(path).mkdir(exist_ok=True)
    with open(os.path.join(path, f'multitask.model'), 'wb+') as f:
        if torch.cuda.device_count() > 1:
            torch.save({
                'model_state_dict': model.model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f)
        else:
            torch.save({
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f)


def train(epoch, batch_size, locations, data_types, task_types):
    datasets = []
    i = 0
    for location, data_type in zip(locations, data_types):
        datasets.append(AMRPenmanMultiTasks(location, task_types[i]))

    data_set = CombineData(datasets)

    train_set = DataLoader(
        data_set, batch_size=batch_size, num_workers=4, shuffle=True
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0
    net.train()
    n = 1
    for source, target in pbar:
        net.zero_grad()

        # text to amr
        loss = net(source, target)
        loss.backward()
        optimizer.step()
        moving_loss += loss.cpu().item()

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Moving Loss: {:.5f};'.format(
                epoch + 1, loss.cpu().item(), moving_loss / n
            )
        )
        n += 1

        if args.pickle_steps > 0 and n % args.pickle_steps == 0:
            print("Saving model to", args.output)
            save(optimizer, net, args.output)


def valid(epoch, batch_size, root, num_beams, num_ret_seq, output_dir=None, prefix='dev'):
    silence_penman()
    data_set = AMRPenman(root)

    valid_set = DataLoader(
        data_set, batch_size=2 * batch_size, num_workers=4, shuffle=False
    )
    dataset = iter(valid_set)
    avg_loss = 0.0
    n = 0.0
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
            loss = net(add_prefix(sentence, AMR_GENERATION), amr_seq)
            avg_loss += loss.cpu().item()
            n += 1
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
            predict_amrs = ps
            amr_predictions += predict_amrs

    num_none = len([p for p in amr_predictions if p is None])
    print("Number of None:", num_none)

    if output_dir is not None:
        ref_out_fn = prefix + '.txt.reference'
        gen_out_fn = prefix + '.txt.generated'
        ref_fname = os.path.join(output_dir, ref_out_fn)
        gen_fname = os.path.join(output_dir, gen_out_fn)
        f_ref = open(ref_fname, 'w')
        f_gen = open(gen_fname, 'w')
        print('Saving %s and %s' % (ref_fname, gen_fname))
        skipped = 0
        for ref_graph, gen_graph in zip(amrs, amr_predictions):
            if gen_graph is None:
                skipped += 1
                continue
            f_ref.write(ref_graph + '\n\n')
            f_gen.write(gen_graph + '\n\n')
        f_ref.close()
        f_gen.close()
        print('Out of %d graphs, skipped %d that did not deserialize properly.' % (len(amrs), skipped))
        print()
        gold_entries = get_entries(ref_fname)
        test_entries = get_entries(gen_fname)
        precision, recall, f_score = compute_smatch(test_entries, gold_entries)
        print(epoch + 1, ' SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
    else:
        precision, recall, f_score = None, None, None

    print(
        'Validation Epoch: {}; Avg Loss: {:.5f};'.format(
            epoch + 1, avg_loss / n
        )
    )

    return avg_loss / n, -f_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMR training')
    parser.add_argument('-t', '--train', type=str, help='Root directory with the training datasets', required=True)
    parser.add_argument('-v', '--validation', type=str, help='Root directory with the validation datasets',  required=True)
    parser.add_argument('-r', '--report_test', type=str, help='Root directory with the test datasets', required=True)
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='The number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=4, help='Mini-batch size')
    parser.add_argument('-o', '--output', type=str, default='./',
                        help='Output folder where the model, and output files will be pickled')
    parser.add_argument('-m', '--model', type=str, default='t5-base', help='model name')
    parser.add_argument('--max_source_length', type=int, default=16, help='Max source length')
    parser.add_argument('--max_target_length', type=int, default=16, help='Max target length')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Checkpoint model')
    parser.add_argument('-p', '--pickle_steps', type=int, default=-1,
                        help='Save the model after every pickle_steps of mini-batches')
    parser.add_argument('--num_beams', type=int, default=5, help='Number of beams used during inference')
    parser.add_argument('--num_ret_seq', type=int, default=5,
                        help='Number of return sequences for each prediction during inference')
    parser.add_argument('--model_type', type=str, default='t5', help='Model type: bart or t5')
    parser.add_argument('--data_type', default="blinkify", help='Type of datasets')
    parser.add_argument('--task_type', default="text2amr", help='Task name')
    parser.add_argument('--val_from_epoch', type=int, default=3, help='Only validation from epochs')
    parser.add_argument('--do_val', default='yes', help='Whether do validation')
    parser.add_argument('--early_termination', default=7, type=int, help='Number of epochs before early termination')
    parser.add_argument('--random_seed', default=0, type=int, help='Default random seed')

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    net = LG(args.model,
             max_source_length=args.max_source_length,
             max_target_length=args.max_target_length,
             model_type=args.model_type)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.checkpoint is not None:
        print("Load model from ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        if torch.cuda.device_count() <= 1:
            net.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.model.module.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.train()

    best_so_far = None
    early_termination = 0
    val_from_epoch = args.val_from_epoch
    for epoch in range(args.epochs):
        train(epoch, args.batch, args.train.split(), args.data_type.split(), args.task_type.split())
        if args.do_val == 'yes':
            if epoch > val_from_epoch:
                print("Validation")
                _, score = valid(epoch,
                                 args.batch,
                                 args.validation,
                                 args.num_beams,
                                 args.num_ret_seq,
                                 output_dir=args.output,
                                 prefix='dev_')
                if best_so_far is None:
                    best_so_far = score
                    print("Saving model to", args.output)
                    save(optimizer, net, args.output)
                elif best_so_far > score:
                    print('Validation score has improved from', best_so_far, ' to ', score)
                    best_so_far = score
                    early_termination = 0
                    print("Saving model to", args.output)
                    save(optimizer, net, args.output)
                    valid(epoch,
                          args.batch,
                          args.report_test,
                          args.num_beams,
                          args.num_ret_seq,
                          args.output,
                          prefix='test_')
                else:
                    early_termination += 1
                if early_termination > args.early_termination:
                    break

    if args.do_val != 'yes':
        save(optimizer, net, args.output)
