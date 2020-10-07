DATA_PATH = "/data/mjsumpter/relation_extraction"

# ---------------------- PATH ----------------------
GREC_DIR = "%s/grec" % DATA_PATH

GREC_JSON_DIR = "%s/json" % GREC_DIR
GREC_RDF_DIR = "%s/rdf" % GREC_DIR
GREC_GRAPH_DIR = "%s/graph" % GREC_DIR

# ---------------------- GRAPHS ----------------------
# Not sure if needed - changing tagging methods

# ---------------------- EMBEDDINGS ----------------------
N2V_DIR = "%s/node2vec" % GREC_DIR
N2V_MODEL = "%s/models" % N2V_DIR
N2V_EMBEDDING = "%s/embeddings" % N2V_DIR
N2V_TEMP = "%s/temp" % N2V_DIR

# ---------------------- SHORTEST PATHS (FEATURES) ----------------------
SP_DIR = "%s/shortest_path" % GREC_DIR
SP_FULL = "%s/"

# ---------------------- API Keys ----------------------
API_KEYS = "%s/api_keys" % DATA_PATH
FRED_LMTD = "%s/fred_key_lmtd" % API_KEYS
FRED_UNLMTD = "%s/fred_key_unlmt" % API_KEYS
