import logging
import torch
from amrlib.models.parse_t5.penman_serializer import PenmanDeSerializer

logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def parse_sentences(self, sentences, num_beams, num_ret_seq):
        # Generate
        graphs_generated = self.model.generate(sentences, num_beams=num_beams, num_ret_seq=num_ret_seq)

        # Extract the top result that isn't clipped and will deserialize
        # At this point "graphs_generated" and "clips" have num_ret_seq for each sent * len(sentences)
        graphs_final = [None] * len(sentences)
        for snum in range(len(sentences)):
            raw_graphs = graphs_generated[snum * num_ret_seq:(snum + 1) * num_ret_seq]
            for bnum, g in enumerate(raw_graphs):
                gstring = PenmanDeSerializer(g).get_graph_string()
                if gstring is not None:
                    graphs_final[snum] = gstring
                    break  # stop deserializing candidates when we find a good one
                else:
                    logger.error('Failed to deserialize, snum=%d, beam=%d' % (snum, bnum))

        # graphs_final = ['# ::snt %s\n%s' % (s, g) if g is not None else None for s, g in zip(sentences, graphs_final)]
        return graphs_final