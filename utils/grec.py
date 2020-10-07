import json
import os
import pathlib
import numpy as np

def get_snippet(json_obj) -> str:
    '''
    Extract snippet from google relation extraction corpus line
    '''
    return json_obj['evidences'][0]['snippet']

def get_metrics(json_file: str, relation: str=None, range: float=0.5) -> dict:
    '''
    When passed a grec json_file, returns a dictionary of metrics for that file.

    range: the +/- range of standard deviation to report min and max w/o outliers
    '''

    metrics = dict()
    metrics['description'] = "Metrics about the snippets of this GREC relation"

    if relation:
        metrics['relation'] = relation
    else:
        metrics['relation'] = pathlib.Path(json_file).stem

    snip_lengths = list()
    with open(json_file) as f:
        relations = json.loads(f.read())
        metrics['num_relations'] = len(relations)

        for relation in relations:
            for evidence in relation['evidences']:
                snip_lengths.append(len(evidence['snippet'].split()))

    elements = np.array(snip_lengths)
    mean = np.mean(elements, axis=0)
    std = np.std(elements, axis=0)

    metrics['mean'] = mean
    metrics['std_dev'] = std
    metrics['max'] = max(snip_lengths)
    metrics['min'] = min(snip_lengths)

    elements_within = [x for x in snip_lengths if (x > mean - range*std)]
    elements_within = [x for x in elements_within if (x < mean + range*std)]

    metrics['max_no_outlier'] = max(elements_within)
    metrics['min_no_outlier'] = min(elements_within)

    return metrics

def json_relation_tag(json_file: str) -> str:
    '''
    Given absolute path to json_file, returns the relation
    '''
    return json_file.split('/')[-1].split('-')[0]
