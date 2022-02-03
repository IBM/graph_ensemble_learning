# Default set of tags to keep when annotatating the AMR.  Throw all others away
# To keep all, redefine this to None
import gzip
import json
import os

import penman

keep_tags = set(['id', 'snt'])


def load_amr_entries(fname, strip_comments=True):
    if fname.endswith('.gz'):
        with gzip.open(fname, 'rb') as f:
            data = f.read().decode()
    else:
        with open(fname, "rt") as f:
            data = f.read()
    # Strip off non-amr header info (see start of Little Prince corpus)
    if strip_comments:
        lines = [l.replace('\x85', ' ') for l in data.split("\n") if not (l.startswith('#') and not \
            l.startswith('# ::'))]

        data = '\n'.join(lines)
    entries = data.split('\n\n')  # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]  # remove any empty entries
    return entries


# Annotate a file with multiple AMR entries and save it to the specified location
def annotate_file(indir, infn, outdir, outfn, spacy_nlp):
    graphs = []
    inpath = os.path.join(indir, infn)
    entries = load_amr_entries(inpath)
    for i in range(len(entries)):
        pen = _process_entry(entries[i], spacy_nlp)
        graphs.append(pen)

    infn = infn[:-3] if infn.endswith('.gz') else infn  # strip .gz if needed
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)


# Worker process that takes in an amr string and returns a penman graph object
# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
# Keep only tags in "keep_tags"
def _process_entry(entry, spacy_nlp, tokens=None):
    pen = penman.decode(entry)  # standard de-inverting penman loading process
    return _process_penman(pen, spacy_nlp, tokens)


# Split out the _process_entry for instances where the string is already converted to a penman graph
def _process_penman(pen, spacy_nlp, tokens=None):
    # Filter out old tags and add the tags from SpaCy parse
    global keep_tags
    if keep_tags is not None:
        pen.metadata = {k: v for k, v in pen.metadata.items() if k in keep_tags}  # filter extra tags
    # If tokens aren't supplied then annoate the graph
    if not tokens:
        tokens = spacy_nlp(pen.metadata['snt'])
    pen.metadata['tokens'] = json.dumps([t.text for t in tokens])
    ner_tags = [t.ent_type_ if t.ent_type_ else 'O' for t in tokens]  # replace empty with 'O'
    pen.metadata['ner_tags'] = json.dumps(ner_tags)
    pen.metadata['ner_iob'] = json.dumps([t.ent_iob_ for t in tokens])
    pen.metadata['pos_tags'] = json.dumps([t.tag_ for t in tokens])
    # Create lemmas
    # The spaCy 2.0 lemmatizer returns -PRON- for pronouns so strip these (spaCy 3.x does not do this)
    # Don't try to lemmatize any named-entities or proper nouns.  Lower-case any other words.
    lemmas = []
    for t in tokens:
        if t.lemma_ == '-PRON-':  # spaCy 2.x only
            lemma = t.text.lower()
        elif t.tag_.startswith('NNP') or t.ent_type_ not in ('', 'O'):
            lemma = t.text
        else:
            lemma = t.lemma_.lower()
        lemmas.append(lemma)
    pen.metadata['lemmas'] = json.dumps(lemmas)
    return pen


# Spacy NLP - lazy loader
def load_spacy(model_name=None):
    import spacy
    spacy_nlp = spacy.load(model_name)
    return spacy_nlp
