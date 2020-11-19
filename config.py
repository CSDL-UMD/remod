import os

DATA_PATH = "/data/mjsumpter/relation_extraction"
SRC_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------------------- PATH ----------------------

JSON_DIR = "%s/json" % DATA_PATH
RDF_DIR = "%s/rdf" % DATA_PATH
GRAPH_DIR = "%s/graph" % DATA_PATH

# ---------------------- EMBEDDINGS ----------------------
N2V_DIR = "%s/node2vec" % DATA_PATH
N2V_MODEL = "%s/models" % N2V_DIR
N2V_EMBEDDING = "%s/embeddings" % N2V_DIR
N2V_TEMP = "%s/temp" % N2V_DIR

# ---------------------- SHORTEST PATHS (FEATURES) ----------------------
SP_DIR = "%s/shortest_path" % DATA_PATH
SP_NODES = "%s/nodes" % SP_DIR
SP_SPLITS_DIR = "%s/splits" % SP_DIR
SP_TRAIN = "%s/train" % SP_SPLITS_DIR
SP_VALID = "%s/valid" % SP_SPLITS_DIR
SP_TEST = "%s/test" % SP_SPLITS_DIR

#---------- MODELS ----------------
MODEL_DIR = "%s/models" % DATA_PATH
TRAIN_LOGS = "%s/logs" % MODEL_DIR

#---------- CLAIMREVIEW ----------------
CLAIM_DIR = "%s/claimreview" % DATA_PATH
CLAIM_DF = "%s/claims.pkl" % CLAIM_DIR

# ---------------------- API Keys ----------------------
API_KEYS = "%s/api_keys" % DATA_PATH
FRED_LMTD = "%s/fred_key_lmtd" % API_KEYS
FRED_UNLMTD = "%s/fred_key_unlmt" % API_KEYS

# ------------- Other Params -------------------
RANDOM_SEED = 2020