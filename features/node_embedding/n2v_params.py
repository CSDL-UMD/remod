STRUCT = {
    "dimensions": 256,
    "walk_length": 50,
    "num_walks": 200,
    "p": 0.25,
    "q": 1,
    "workers": 32,
    "window": 15,
    "min_count": 1,
}

LOCAL = {
    "dimensions": 256,
    "walk_length": 50,
    "num_walks": 200,
    "p": 1,
    "q": 0.5,
    "workers": 32,
    "window": 15,
    "min_count": 1,
}

# Custom parameters can be added here
CUSTOM = {
    # "dimensions": ,
    # "walk_length": ,
    # "num_walks": ,
    # "p": ,
    # "q": ,
    # "workers": ,
    # "window": ,
    # "min_count": ,
}


n2v_param_dict = {"struct": STRUCT, "local": LOCAL, "custom": CUSTOM}

