# Graph ensemble for AMR

from random import shuffle
import argparse
import penman
from amrlib.evaluate.smatch_enhanced import compute_smatch
from ensemble.utils import match_pair, align, get_entries
import re
from penman.model import Model
import time
import warnings

model = Model()

warnings.filterwarnings("ignore")


def get_node_maps(best_mapping):
    mab = {}
    mba = {}
    i = 0
    for j in best_mapping:
        mab['a' + str(i)] = 'b' + str(j)
        mba['b' + str(j)] = 'a' + str(i)
        i += 1
    return mab, mba


def get_attribute(best_mapping, attributes1, attributes2):
    mab, mba = get_node_maps(best_mapping)
    a1 = {}
    a2 = {}
    for a in attributes1:
        a1[a] = 1
    for a in attributes2:
        a2[a] = 1

    # attributes in the first list that present in the second list as well
    i1 = []
    # attributes in the first but not in the second list
    r1 = []
    for a in attributes1:
        r = a[1]
        if r in mab:
            if (a[0], mab[r], a[2]) not in attributes2:
                r1.append(a)
            else:
                i1.append(a)
        else:
            r1.append(a)

    # attributes in the second list that present in the first list as well
    i2 = []
    # attributes in the second but not in the first list
    r2 = []
    for a in attributes2:
        r = a[1]
        if r in mba:
            if (a[0], mba[r], a[2]) not in attributes1:
                r2.append(a)
            else:
                i2.append(a)
        else:
            r2.append(a)
    return r1, r2, i1, i2


def get_instance(best_mapping, instance1, instance2):
    mab, mba = get_node_maps(best_mapping)
    a1 = {}
    a2 = {}
    for a in instance1:
        a1[a] = 1
    for a in instance2:
        a2[a] = 1

    # instances in the first list that present in the second list as well
    i1 = []
    # instances in the first but not in the second list
    r1 = []
    for a in instance1:
        r = a[1]
        if r in mab:
            if (a[0], mab[r], a[2]) not in instance2:
                r1.append(a)
            else:
                i1.append(a)
        else:
            r1.append(a)

    # attributes in the second list that present in the first list as well
    i2 = []
    # attributes in the second but not in the first list
    r2 = []
    for a in instance2:
        r = a[1]
        if r in mba:
            if (a[0], mba[r], a[2]) not in instance1:
                r2.append(a)
            else:
                i2.append(a)
        else:
            r2.append(a)
    return r1, r2, i1, i2


def get_relation(best_mapping, relation1, relation2):
    mab, mba = get_node_maps(best_mapping)
    a1 = {}
    a2 = {}
    for a in relation1:
        a1[a] = 1
    for a in relation2:
        a2[a] = 1

    # relations in the first list that present in the second list as well
    i1 = []
    # relations in the first but not in the second list
    r1 = []
    for a in relation1:
        if a[1] in mab and a[2] in mab:
            if (a[0], mab[a[1]], mab[a[2]]) not in relation2:
                r1.append(a)
            else:
                i1.append(a)
        else:
            r1.append(a)

    # relations in the second list that present in the first list as well
    i2 = []
    # relations in the second but not in the first list
    r2 = []
    for a in relation2:
        if a[1] in mba and a[2] in mba:
            if (a[0], mba[a[1]], mba[a[2]]) not in relation1:
                r2.append(a)
            else:
                i2.append(a)
        else:
            r2.append(a)
    return r1, r2, i1, i2


def get_map(best_mapping, instance1, attributes1, relation1, instance2, attributes2, relation2):
    a1, a2, ai1, ai2 = get_attribute(best_mapping, attributes1, attributes2)
    i1, i2, ii1, ii2 = get_instance(best_mapping, instance1, instance2)
    r1, r2, ri1, ri2 = get_relation(best_mapping, relation1, relation2)
    return a1, ai1, i1, ii1, r1, ri1, a2, ai2, i2, ii2, r2, ri2


def get_variables(g):
    v = {}
    for t in g.triples:
        if t[1] == ':instance':
            v[t[0]] = 1
    return v


def get_triples(amr_str):
    g = penman.decode(amr_str)
    variables = get_variables(g)
    instances_dict = {}
    attributes_dict = {}
    relations_dict = {}
    for t in g.triples:
        if t[1] == ':instance':
            instances_dict[t] = 1
        elif t[0] in variables and t[2] in variables:
            relations_dict[t] = 1
        else:
            attributes_dict[t] = 1

    return instances_dict, attributes_dict, relations_dict


def match_count(amr1, amr2):
    instances = {}
    attributes = {}
    relations = {}
    best_mapping, instance1, attributes1, relation1, instance2, attributes2, relation2, node_maps_1, node_maps_2 = match_pair(
        (amr1, amr2))
    a1, ai1, i1, ii1, r1, ri1, a2, ai2, i2, ii2, r2, ri2 = get_map(best_mapping, instance1, attributes1, relation1,
                                                                   instance2, attributes2, relation2)

    mab, mba = get_node_maps(best_mapping)

    # common in both amr
    for a in ai1:
        attributes[(a[0], node_maps_1[a[1]], a[2])] = 1

    for a in ii1:
        instances[(a[0], node_maps_1[a[1]], a[2])] = 1

    for a in ri1:
        relations[(a[0], node_maps_1[a[1]], node_maps_1[a[2]])] = 1

    # exist in the second but not in the first amr
    for a in a2:
        n = a[1]
        if n in mba:
            attributes[(a[0], node_maps_1[mba[n]], a[2])] = 1

    for a in i2:
        n = a[1]
        if n in mba:
            instances[(a[0], node_maps_1[mba[n]], a[2])] = 1

    for a in r2:
        n1 = a[1]
        n2 = a[2]
        if n1 in mba and n2 in mba:
            relations[(a[0], node_maps_1[mba[n1]], node_maps_1[mba[n2]])] = 1

    convert_instances = {}
    for k, v in instances.items():
        convert_instances[convert_instance(k)] = v

    convert_attributes = {}
    for k, v in attributes.items():
        convert_attributes[convert_attribute(k)] = v

    convert_relations = {}
    for k, v in relations.items():
        convert_relations[convert_relation(k)] = v
    return convert_instances, convert_attributes, convert_relations


def convert_instance(i):
    t = (i[1], ':' + i[0], i[2])
    return t


def convert_relation(i):
    t = (i[1], ':' + i[0], i[2])
    t = model.deinvert(t)
    return t


def convert_attribute(i):
    a = i[2]
    if a[-1] == '_':
        a = a[:-1]
        a = '"' + a + '"'
    t = (i[1], ':' + i[0], a)
    return t


def ensemble(amrs, threshold):
    amr1 = amrs[0]
    instances = {}
    attributes = {}
    relations = {}
    try:
        instances, attributes, relations = get_triples(amr1)
        for amr2 in amrs[1:]:
            i, a, r = match_count(amr1, amr2)
            for k, v in i.items():
                if k in instances:
                    instances[k] += 1
                else:
                    instances[k] = 1

            for k, v in a.items():
                if k in attributes:
                    attributes[k] += 1
                else:
                    attributes[k] = 1

            for k, v in r.items():
                if k in relations:
                    relations[k] += 1
                else:
                    relations[k] = 1
            n = len(amrs)

        original_g = penman.decode(amr1)
        g = penman.decode(amr1)
        # update the triples
        update_instances(instances, threshold, n, g)
        update_attributes(attributes, threshold, n, g)
        update_relations(relations, threshold, n, g)
    except:
        for amr in amrs:
            try:
                g = penman.decode(amr)
                original_g = penman.decode(amr)
            except:
                g = None
                original_g = None
        if g is None:
            g = penman.decode('(z / end-01)')
        if original_g is None:
            original_g = penman.decode('(z / end-01)')
    deduplicate_triples(g)
    freq = threshold * len(amrs)
    ensemble_support = get_total_support(instances, attributes, relations, g, freq)
    original_support = get_total_support(instances, attributes, relations, original_g, freq)
    return g, ensemble_support, original_support


def deduplicate_triples(g):
    triples = {}
    clean_triples = []
    for t in g.triples:
        rt = inverse(t)
        if t in triples or rt in triples:
            pass
        else:
            triples[t] = 1
            clean_triples.append(t)
    g.triples = clean_triples


def get_total_support(instances, attributes, relations, g, freq):
    r = 0.0
    n = 0
    for t in g.triples:
        if t in instances:
            if instances[t] >= freq:
                r += instances[t] - 1
                n += 1
        elif t in attributes:
            if attributes[t] >= freq:
                r += attributes[t] - 1
                n += 1
        else:
            c = count_relation_freq(relations, t)
            if t in relations and c >= freq:
                r += c - 1
                n += 1
    if n == 0:
        n = 1
    return r
    # return r/n


def update_instances(instances, threshold, n, g):
    # only keep the instances with votes greater than the threshold
    qualified_instance = {}
    for t, v in instances.items():
        if v / (n + 0.0) >= threshold:
            qualified_instance[t] = v

    # keep instances with the major votes if there exist multiple instances with the same instance relations
    r = {}
    for t, v in qualified_instance.items():
        p = (t[0], t[1])
        if p not in r:
            r[p] = (t, v)
        elif r[p][1] < v:
            r[p] = (t, v)

    triples = []
    for t in g.triples:
        p = (t[0], t[1])
        if p in r:
            # keep the instance with the greatest votes
            triples.append(r[p][0])
        elif t in instances:
            # keep instance definition even not sufficient votes to make sure that the graph is a connected graph
            triples.append(t)
        else:
            # non-instance triples, keep all of them
            triples.append(t)
    g.triples = triples


def update_attributes(attributes, threshold, n, g):
    # only keep the attributes with votes greater than the threshold
    qualified_attributes = {}
    for t, v in attributes.items():
        if v / (n + 0.0) >= threshold:
            qualified_attributes[t] = v

    # keep attributes with the major votes if there exist multiple attributes with the same attribute relations
    r = {}
    for t, v in qualified_attributes.items():
        p = (t[0], t[1])
        if p not in r:
            r[p] = (t, v)
        elif r[p][1] < v:
            r[p] = (t, v)

    triples = []
    for t in g.triples:
        p = (t[0], t[1])
        if p in r and t in attributes:
            # keep the attribute with the greatest votes
            if r[p][0] not in triples:
                triples.append(r[p][0])
        elif t in attributes:
            # ignored due to insufficient votes
            pass
        else:
            # non-attributes triples, keep all of them
            triples.append(t)

    triples_dict = {}
    for t in g.triples:
        triples_dict[t] = 1

    for t in triples:
        triples_dict[t] = 1

    # add new attributes
    for t, v in qualified_attributes.items():
        if t[1] != ':TOP':
            if t not in triples_dict:
                triples.append(t)
    g.triples = triples


def count_relation_freq(relations, t):
    r = 0
    if t in relations:
        r = relations[t]
    rt = inverse(t)
    if rt in relations:
        r += relations[rt]
    return r


def update_relations(relations, threshold, n, g):
    # only keep the relations with votes greater than the threshold
    qualified_relations = {}
    for t, v in relations.items():
        c = count_relation_freq(relations, t)
        if c / (n + 0.0) >= threshold:
            qualified_relations[t] = v

    # keep relations with the major votes if there exist multiple relations between variables
    r = {}
    for t, v in qualified_relations.items():
        p = (t[0], t[2])
        if p not in r:
            r[p] = (t, v)
        elif r[p][1] < v:
            r[p] = (t, v)

    triples_dict = {}
    for t in g.triples:
        triples_dict[t] = 1

    # remove non-qualified relations
    triples = []
    for t in g.triples:
        if t in relations:
            p = (t[0], t[2])
            if p in r:
                if t == r[p][0]:
                    triples.append(t)
                else:
                    if r[p][0] not in triples_dict and inverse(r[p][0]) not in triples_dict:
                        # replace t by r[p][0] because it has more votes
                        triples.append(r[p][0])
                    else:
                        pass
            else:
                triples.append(t)
        else:
            triples.append(t)

    for t in triples:
        triples_dict[t] = 1

    # add new relations
    for t, v in qualified_relations.items():
        if t not in triples_dict and inverse(t) not in triples_dict:
            if not invalid(t):
                triples.append(t)

    k = 0
    for r in triples:
        if ('h2', ':degree', 'e2') == r:
            k += 1
    g.triples = triples


def invalid(t):
    if t[1].startswith(':snt'):
        return True
    else:
        return False


def inverse(t):
    r = t[1]
    if r.endswith('-of'):
        return (t[2], r[:-3], t[0])
    elif r == ':mod':
        return (t[2], ':domain', t[0])
    elif r == ':domain':
        return (t[2], ':mod', t[0])
    else:
        return (t[2], r + '-of', t[0])


def get_entry(e):
    lines = [l.strip() for l in e.splitlines()]
    lines = [l for l in lines if (l and not l.startswith('#'))]
    string = ' '.join(lines)
    string = string.replace('\t', ' ')  # replace tabs with a space
    string = re.sub(' +', ' ', string)  # squeeze multiple spaces into a single
    return string


def vote(test_entries, gold_entries, threshold):
    i = 0
    ensemble_entries = []
    scores = []
    original_scores = []
    for j in range(len(test_entries[0])):
        amrs = [test_entries[l][j] for l in range(len(test_entries))]
        g, score, original_score = ensemble(amrs, threshold=threshold)
        scores.append(score)
        original_scores.append(original_score)
        try:
            s = penman.encode(g)
            penman.decode(s)
            get_entry(s)
            ensemble_entries.append(get_entry(s))
        except:
            i += 1
            ensemble_entries.append(amrs[0])
    precision, recall, f_score = compute_smatch(ensemble_entries, gold_entries)
    print(' SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
    return ensemble_entries, scores, precision, recall, f_score, original_scores


def graphene(test_entries, gold_entries, threshold, ties_broken_arbitrarily='no'):
    all_ensembles = []
    all_scores = []
    for i in range(len(test_entries)):
        print("Pivot iteration:", i, "/", len(test_entries)-1)
        voters = [test_entries[i]]
        for j in range(len(test_entries)):
            if i != j:
                voters.append(test_entries[j])
        ensemble_entries, scores, _, _, _, _ = vote(voters, gold_entries, threshold)
        print(sum(scores) / len(scores))
        all_ensembles.append(ensemble_entries)
        all_scores.append(scores)

    best_entries = []
    best_score = []
    votes = [0.0 for c in all_ensembles]
    for i in range(len(all_ensembles[0])):
        bs = None
        idx = None
        best_indices = []
        for j in range(len(all_ensembles)):
            if bs is None:
                bs = all_scores[j][i]
            elif bs < all_scores[j][i]:
                bs = all_scores[j][i]
        for j in range(len(all_ensembles)):
            if bs == all_scores[j][i]:
                best_indices.append(j)

        if ties_broken_arbitrarily == 'yes':
            shuffle(best_indices)

        idx = best_indices[0]
        votes[idx] += 10
        best_entries.append(all_ensembles[idx][i])
        best_score.append(bs)
    print("Avg score", sum(best_score) / len(best_score))
    precision, recall, f_score = compute_smatch(best_entries, gold_entries)
    best_score = sum(best_score) / len(best_score)
    votes = [v / len(all_ensembles[0]) for v in votes]
    print("Votes", votes)
    return best_entries, best_score, precision, recall, f_score


def graphene_smatch(test_entries, gold_entries, threshold):
    # create pivot graphs
    all_ensembles = []
    all_scores = []
    for i in range(len(test_entries)):
        print("Pivot iteration:", i, "/", len(test_entries)-1)
        voters = [test_entries[i]]
        for j in range(len(test_entries)):
            if i != j:
                voters.append(test_entries[j])
        ensemble_entries, scores, _, _, _, _ = vote(voters, gold_entries, threshold)
        print(sum(scores) / len(scores))
        all_ensembles.append(ensemble_entries)
        all_scores.append(scores)

    for i in range(len(test_entries)):
        all_ensembles.append(test_entries[i])

    # choose best graph based on best average smatch
    best_entries = []
    for n in range(len(all_ensembles[0])):
        best_f_score = None
        best_graph = None
        for i in range(len(all_ensembles)):
            graphs = [all_ensembles[i][n]] * len(test_entries)
            others = []
            for j in range(len(test_entries)):
                others.append(test_entries[j][n])
            precision, recall, f_score = compute_smatch(graphs, others)
            if best_f_score is None or best_f_score < f_score:
                best_f_score = f_score
                best_graph = all_ensembles[i][n]
        best_entries.append(best_graph)
        if n % 10 == 0:
            print(n, len(test_entries[0]))

    precision, recall, f_score = compute_smatch(best_entries, gold_entries)
    return best_entries, f_score, precision, recall, f_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Ensemble (Graphene) for AMR parsing')

    parser.add_argument(
        '-a', '--algorithm',
        default='graphene_smatch',
        type=str,
        help='graphene, graphene_smatch, vote. In the vote option, the first graph in the input is chosen as the pivot graph. In the '
             'graphene option, every input graph is chosen as a pivot graph once and the best among the modified pivot'
             'graphs is chosen as the final prediction. Graphene_smatch is similar to graphene except in the last step, '
             'the best modified pivot graph was chosen based on Smatch rather than based on support similar to Barzdins et al.')

    parser.add_argument(
        '--data',
        default='../data/lp_SPRING.txt.wiki ../data/lp_SPRING_703.txt.wiki '
                '../data/lp_SPRING_803.txt.wiki ../data/lp_SPRING_903.txt.wiki '
                '../data/lp_t5.txt.wiki ../data/lp_APT.txt ../data/lp_cai_lam.wiki',
        type=str,
        help='Paths to prediction files')

    parser.add_argument(
        '--gold',
        default='../data/spring_gold_lp.txt',
        type=str,
        help='Path to the gold prediction file, needed to calculate the SMATCH statistics. If it does not exists, '
             'simply use a dummy gold file like the input file itself.')

    parser.add_argument(
        '-t', '--theta',
        default=0.5,
        type=float,
        help='Minimum support threshold, can be tuned to get better results if there is an independent dev set.')

    parser.add_argument(
        '--align',
        default='yes',
        type=str,
        help='Whether to align with the gold file using the id tag. When evaluating the models on different datasets,'
             ' the prediction files may not contain the amrs in the same order, or there are some AMRs that missing. '
             'We need to align them with the same order to make sure that the ensemble working on correct AMR '
             'prediction and the SMATCH statistics is correct. ')

    parser.add_argument(
        '--ties_broken_arbitrarily',
        default='yes',
        type=str,
        help='When ties in support happen, whether choose the ensemble graphs randomly, '
             'otherwise choose the first ensemble graph in the list.')

    args = parser.parse_args()

    start = time.time()
    data = args.data.split()
    args = parser.parse_args()

    ref_fname = args.gold

    print('Gold file:', ref_fname)
    print(data)
    gen_fname = data
    original_gold_entries, gold_entries = get_entries(ref_fname)
    test_entries = []
    for gen_file in gen_fname:
        print(gen_file)
        original_test_entries_1, test_entries_1 = get_entries(gen_file)
        if args.align == 'no':
            # If the AMRs in the prediction file  are not in the same order with the AMRs in the gold fil
            # we need to align the test graphs with gold graphs via ID tag in order to make sure that
            # the test entries contain the same number of amrs and in the same order as the gold entries
            # This ensures that the SMATCH statistics output is correct.
            test = align(original_gold_entries, original_test_entries_1, test_entries_1)
        else:
            test = test_entries_1
        test_entries.append(test)

    for test in test_entries:
        precision, recall, f_score = compute_smatch(test, gold_entries)
        print(' SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))

    theta = args.theta
    if args.algorithm == 'graphene':
        ensemble_entries, scores, precision, recall, f_score = graphene(test_entries,
                                                                        gold_entries,
                                                                        theta,
                                                                        args.ties_broken_arbitrarily)
    elif args.algorithm == 'graphene_smatch':
        ensemble_entries, scores, precision, recall, f_score = graphene_smatch(test_entries,
                                                                               gold_entries,
                                                                               theta)
    else:
        ensemble_entries, scores, precision, recall, f_score, original_scores = vote(test_entries,
                                                                                     gold_entries,
                                                                                     theta)
        scores = sum(scores) / len(scores)
        original_scores = sum(original_scores) / len(original_scores)
        print("Scores", scores, original_scores)

    print('Best score so far', scores, f_score)
    best_prediction = [penman.encode(penman.decode(g)) for g in ensemble_entries]

    outputs = []
    for g, p in zip(original_gold_entries, best_prediction):
        r = penman.decode(g)
        s = ''
        if 'snt' in r.metadata:
            s += '# ::snt ' + r.metadata['snt'] + '\n'
        if 'id' in r.metadata:
            s += '# ::id ' + r.metadata['id'] + '\n'
        if 'tok' in r.metadata:
            s += '# ::tok ' + r.metadata['tok'] + '\n'
        s += p
        outputs.append(s)

    print("Average running time per graph:", (time.time() - start) / len(original_gold_entries),
          len(original_gold_entries))
    output_file = args.algorithm + '.txt'
    with open(output_file, 'wt') as f:
        print('Write prediction to', output_file)
        f.write('\n\n'.join(map(str, outputs)))
