STRUCT = {
    "dimensions": 256,
    "walk_length": 50,
    "num_walks": 200,
    "p": 2,
    "q": 3,
    "workers": 32,
    "window": 15,
    "min_count": 1,
}

LOCAL = {
    "dimensions": 256,
    "walk_length": 50,
    "num_walks": 200,
    "p": 2,
    "q": 0.5,
    "workers": 32,
    "window": 15,
    "min_count": 1,
}

CUSTOM = {
    # "dimensions": 256,
    # "walk_length": 50,
    # "num_walks": 200,
    # "p": 2,
    # "q": 0.5,
    # "workers": 32,
    # "window": 15,
    # "min_count": 1,
}

n2v_param_dict = {
    "best": STRUCT,
    "local": LOCAL,
    "custom": CUSTOM
}

