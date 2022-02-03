import argparse
import logging

import blink.main_dense as main_dense
import penman
from tqdm import tqdm

from amr_utils.datasets.dataset import AMR_GENERATION, TEXT_GENERATION, SENSE_GENERATION

logger = logging.getLogger(__name__)


# Class for adding :wiki tags to a graph using Blink: https://github.com/facebookresearch/BLINK.
class BlinkWikiAdder:
    def __init__(self, models_path, fast=False):
        # For debug stats
        self.wiki_lookups = 0
        self.wiki_found = 0

        config = {
            "test_entities": None,
            "test_mentions": None,
            "interactive": False,
            "biencoder_model": models_path + "biencoder_wiki_large.bin",
            "biencoder_config": models_path + "biencoder_wiki_large.json",
            "entity_catalogue": models_path + "entity.jsonl",
            "entity_encoding": models_path + "all_entities_large.t7",
            "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
            "crossencoder_config": models_path + "crossencoder_wiki_large.json",
            "top_k": 10,
            "show_url": False,
            "fast": fast,  # set this to be true if speed is a concern
            "output_path": models_path + "logs/",  # logging directory
            "faiss_index": None,  # "flat",
            "index_path": models_path + "faiss_flat_index.pkl",
        }
        self.args_blink = argparse.Namespace(**config)
        self.models = main_dense.load_models(self.args_blink, logger=logger)

    # Print some status
    def get_stat_string(self):
        string = 'Attempted {:,} wiki lookups\n'.format(self.wiki_lookups)
        string += 'For a total of {:,} non-null (ie.. :wiki -) graph updates\n'.format(self.wiki_found)
        return string[:-1]  # string final line-feed

    # Load a file, add wiki attribs and save it
    def wikify_file(self, infn, outfn):
        new_graphs = []
        for graph in tqdm(penman.load(infn)):
            try:
                new_graph = self.wikify_graph(graph)
            except:
                new_graph = graph
            new_graphs.append(new_graph)
        penman.dump(new_graphs, outfn, indent=6)

    # Add a wiki attribute to all nodes with a :name edge and node
    def wikify_graph(self, graph):
        gid = graph.metadata.get('id', '')
        # Check for name attributes.  These shouldn't be present but might.
        for name_attrib in [t for t in graph.attributes() if t.role == ':name']:
            logger.warning('%s has :name attrib in graph %s' % (gid, name_attrib))
        # Find all the name edges and loop through them
        name_edges = [t for t in graph.edges() if t.role == ':name']
        for_blink = []
        sample_id = 0
        for name_edge in name_edges:
            # Get the associated name string and the parent node to add :wiki to
            name_attribs = [t.target for t in graph.attributes() if t.source == name_edge.target]
            name_attribs = [a.replace('"', '') for a in name_attribs]
            name_string = ' '.join(name_attribs)
            # This typically does not occur (only 1 instance in LDC2015E86),
            # however generated graphs may have more
            if not name_string:
                logger.warning('%s No name assosiated with the edge %s' % (gid, str(name_edge)))
                continue
            if 'snt' in graph.metadata:
                sentence = graph.metadata['snt']
            else:
                sentence = graph.metadata['tok']
            sentence = sentence.replace(f"{AMR_GENERATION} ;", "").replace(f"{TEXT_GENERATION} ;", "").replace(
                f"{SENSE_GENERATION} ;", "")
            if name_string.lower() in sentence.lower():
                left = sentence.lower().lower().find(name_string.lower())
                right = left + len(name_string)
            else:
                left = len(sentence)
                right = 0
                print(f"String '{name_string}' not found in sentence: {sentence}, not exact context provided to Blink")
            sample = {
                "id": sample_id,
                "label": "unknown",
                "label_id": -1,
                "context_left": sentence[:left].strip().lower(),
                "mention": name_string.lower(),
                "context_right": sentence[right:].strip().lower(),
            }
            self.wiki_lookups += 1
            sample_id += 1
            for_blink.append(sample)
            # Lookup the phrase in the spotlight datasets.
        if len(for_blink) > 0:
            _, _, _, _, _, predictions, scores, = main_dense.run(self.args_blink, logger, *self.models, test_data=for_blink)
        else:
            predictions = []
        idx = 0
        for name_edge in name_edges:
            name_attribs = [t.target for t in graph.attributes() if t.source == name_edge.target]
            name_attribs = [a.replace('"', '') for a in name_attribs]
            name_string = ' '.join(name_attribs)
            if not name_string:
                logger.warning('%s No name assosiated with the edge %s' % (gid, str(name_edge)))
                continue
            pp = predictions[idx]
            idx += 1
            pp = [p for p in pp if not p.startswith('List of')]
            p = f'"{pp[0]}"' if pp else '-'
            wiki_val = p.replace(' ', '_')
            if wiki_val is not None and wiki_val is not '-':
                self.wiki_found += 1
            else:
                logger.debug('No wiki datasets for %s' % name_string)
                wiki_val = '-'  # Per AMR spec, a dash is used for no reference
            # Find the index of the parent in the graph.triples
            # The index technically doesn't matter but it may impact the print order
            parent_var = name_edge.source
            parent_triples = [t for t in graph.triples if t[1] == ':instance' and t[0] == parent_var]
            if len(parent_triples) != 1:
                logger.error('%s Graph lookup error for %s returned %s' % (gid, parent_var, parent_triples))
                continue
            index = graph.triples.index(parent_triples[0])
            # Now add this to the graph just after the parent and add an empty epidata entry
            triple = (parent_var, ':wiki', wiki_val)
            graph.triples.insert(index, triple)
            graph.epidata[triple] = []
        return graph

    # Get all sentences from an AMR file
    @staticmethod
    def get_sents_from_AMR(infn):
        sents = []
        for graph in penman.load(infn):
            if 'snt' in graph.metadata:
                sents.append(graph.metadata['snt'])
            else:
                sents.append(graph.metadata['tok'])
        return sents
