import glob
import gzip

from amrlib.models.parse_t5.penman_serializer import PenmanSerializer
from torch.utils.data import Dataset
from tqdm import tqdm

AMR_GENERATION = "amr generation"
TEXT_GENERATION = "text generation"
SENSE_GENERATION = "sense generation"


class CombineData(Dataset):
    def __init__(self, datasets):
        self.data = []
        for d in datasets:
            self.data.extend(d.data)
        print("Combine datasets size:", len(self.data))

    def __getitem__(self, index):
        source, target = self.data[index]
        return source, target

    def __len__(self):
        return len(self.data)


class AMRPenman(Dataset):
    def __init__(self, root, ret_type='all'):
        print(root)
        self.data = []
        files = glob.glob(root.replace("'", ""))
        print("Number of file:", len(files))
        amrs = []
        texts = []
        graphs = []
        for file in files:
            entries = load_and_serialize(file, False)
            texts += ['%s' % sent for sent in entries['sents']]
            amrs += ['%s' % graph for graph in entries['serials']]
            graphs += ['%s' % graph for graph in entries['graphs']]

        for i in range(len(amrs)):
            self.data.append((amrs[i], texts[i], graphs[i]))

        print("Data size:", len(self.data))

        self.ret_type = ret_type

    def __getitem__(self, index):
        amr, text, amr_penman = self.data[index]
        if self.ret_type == 'amr':
            return amr
        elif self.ret_type == 'text':
            return text
        else:
            return amr, text, amr_penman

    def __len__(self):
        return len(self.data)


class AMRPenmanMultiTasks(Dataset):

    def __init__(self, root, task_type='all'):
        print(root)
        self.data = []
        files = glob.glob(root.replace("'", ""))
        print("Number of file:", len(files))
        amrs = []
        texts = []
        graphs = []
        for file in files:
            entries = load_and_serialize(file, False)
            texts += ['%s' % sent for sent in entries['sents']]
            amrs += ['%s' % graph for graph in entries['serials']]
            graphs += ['%s' % graph for graph in entries['graphs']]

        for i in range(len(amrs)):
            if task_type == 'all':
                self.data.append((add_prefix_single(amrs[i], TEXT_GENERATION), texts[i]))
                self.data.append((add_prefix_single(texts[i], AMR_GENERATION), amrs[i]))
            elif task_type == 'amr2text':
                self.data.append((add_prefix_single(amrs[i], TEXT_GENERATION), texts[i]))
            elif task_type == 'text2amr':
                self.data.append((add_prefix_single(texts[i], AMR_GENERATION), amrs[i]))

        print("Data size:", len(self.data))

    def __getitem__(self, index):
        amr, text = self.data[index]
        return amr, text

    def __len__(self):
        return len(self.data)


def load_amr_entries(fname, strip_comments=True):
    if fname.endswith('.gz'):
        with gzip.open(fname, 'rb') as f:
            data = f.read().decode()
    else:
        with open(fname, "rt") as f:
            data = f.read()
    # Strip off non-amr header info (see start of Little Prince corpus)
    if strip_comments:
        lines = [l.replace('\x85', ' ') for l in data.split("\n") if not (l.startswith('#')
                                                                          and not l.startswith('# ::'))]
        data = '\n'.join(lines)
    entries = data.split('\n\n')  # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]  # remove any empty entries
    return entries


def load_and_serialize(fpath, progress=True, max_entries=None):
    entries = load_amr_entries(fpath)[:max_entries]
    serials = {'graphs': [], 'sents': [], 'serials': []}
    print('Loading and converting', fpath)
    for entry in tqdm(entries, ncols=100, disable=not progress):
        try:
            serializer = PenmanSerializer(entry)
            if None not in serializer.tokens:
                serials['graphs'].append(entry)
                serials['serials'].append(serializer.get_graph_string())
                serials['sents'].append(serializer.get_meta('snt').strip())
        except Exception as e:
            print('Parse error', entry, e)
    return serials


def add_prefix(sentences, prefix):
    results = [prefix + " ; " + s for s in sentences]
    return results


def add_prefix_single(sentence, prefix):
    result = prefix + " ; " + sentence
    return result
