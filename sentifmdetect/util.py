#!/usr/bin/env python3
'''
util.py
sentifmdetect17 
11/24/17
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import os
import logging.config
import yaml
from sentifmdetect import settings
import collections
from itertools import groupby, chain
import glob
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers() #json pickle can now be used with numpy

def setup_logging(
        default_path='logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def read_json(fp):
    with open(fp, "rt") as f_in:
        data_enc = json.load(f_in)
    print(f"Read file {fp}.")
    data = jsonpickle.decode(data_enc)
    return data

def write_json(data, fp):
    data_enc = jsonpickle.encode(data)
    with open(fp, "wt") as f_out:
        json.dump(data_enc, f_out)
    print(f"Wrote file {fp}.")
    return True

def read_features():

    # check if premade feature file is present
    try:
        with open(settings.FEATURE_OPT_FP, "rt") as feature_in:
            feature_data_enc = json.load(feature_in)
        feature_data = jsonpickle.decode(feature_data_enc)
        return feature_data

    except Exception as e:
        logging.info(f"Could not load json features. {str(e)}")
        return False

def write_features(feature_data):
    """
    Encodes and write feature data.
    :param feature_data:
    """
    # encode numpy to json
    feature_data_enc = jsonpickle.encode(feature_data)

    with open(settings.FEATURE_OPT_FP, "wt") as feature_out:
        json.dump(feature_data_enc, feature_out)

    # test by reading
    with open(settings.FEATURE_OPT_FP, "rt") as feature_in:
        feature_data_enc = json.load(feature_in)
    feature_data_reloaded = jsonpickle.decode(feature_data_enc)

    logging.info(f"Wrote feature data to {settings.FEATURE_OPT_FP}.")

def nested_update(d, u):
    """
    Nested dict update
    :param d: dict to update
    :param u: update
    :return: updated dict
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def read_metadata(fp):
    with open(fp, "rt") as meta_in:
        metadata_enc = json.load(meta_in)
    metadata = jsonpickle.decode(metadata_enc)
    logging.info(f"Read exp data at {fp}")
    return metadata

def write_metadata(update_data):
    """
    Updates the experiment metadata with a dict.
    Then writes the metadata produced by an experiment to the set optdir.
    :param metadata: a dict with the metadata.
    """
    settings.METADATA = nested_update(settings.METADATA, update_data)
    fp = os.path.join(settings.OPT_DIRP, "metadata.json")
    # encode numpy to json
    metadata_enc = jsonpickle.encode(settings.METADATA)
    with open(fp, "wt") as meta_out:
        json.dump(metadata_enc, meta_out)
    logging.info(f"Wrote experiment metadata to {fp}.")

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def remove_duplicates(seq):
    '''
    Removes duplicates fron sequences and maintains order as opposed to set().
    :param seq:
    :return:
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_duplicate_indices(seq):
    '''
    Get the index of duplicates from sequences.
    :param seq:
    :return:
    '''
    seen = set()
    seen_add = seen.add
    return [i for i, x in enumerate(seq) if (x in seen or seen_add(x))]

def get_consecutive_duplicate_indices(seq):
    grouped = [(k, sum(1 for i in g)) for k, g in groupby(seq)]
    # grouped = [(k, list(g)) for k, g in groupby(seq)]
    print(grouped)
    consec_idc = []
    j = 0
    for val, cnt in grouped:
        if cnt > 1:
            dupe_idc = list(range(j+1, j+cnt))
            consec_idc.extend(dupe_idc)
        j += cnt
    return consec_idc

def remove_duplicates_by_index(*lists, index=[]):
    for list in lists:
        yield [x for i, x in enumerate(list) if i not in index]

def multipattern_glob(*patterns):
    return chain.from_iterable(glob.iglob(pattern) for pattern in patterns)

if __name__ == "__main__":
    l = [2, 2, 3, 1, 1, 1, 1, 2, 3, 4, 1, 2, 2, 6, 6, 6, ]
    print(get_consecutive_duplicate_indices(l))
    print(list(remove_duplicates_by_index(l, index=get_consecutive_duplicate_indices(l))))
