# Fact-Checking with Relation Extraction using Node Embeddings of Dependency Trees

From the technical report: [Beyond Claim Matching: Fact-Checking Novel Claims Using Relation Extraction]()

The following are the instructions to strictly reproduce the results cited in the paper.

![Pipeline](figs/pipeline.png)

## Prerequisites
The necessary packages can be installed as an Anaconda environment from the `environment.yml` file.

### Dataset

First, change `DATA_PATH` in `config.py` to the directory that you would like to store your data. Then,

To download the Google Relation Extraction Corpus (GREC):

```
python get_data.py
```

To build a corpus of claims, please consult the documentation for the [Google FactCheck API](https://toolbox.google.com/factcheck/apis), and add your data to the `JSON_DIR` as a JSON in the same format as the GREC.

## Preprocessing

### FRED Parsing
To parse, first acquire a [FRED](http://wit.istc.cnr.it/stlab-tools/fred/) API key, and past it in a text file at `DATA_PATH/api_keys/fred_key_lmtd`. Then:

```
python generate_fred_rdfs.py
```

This should produce a directory of RDF files, one for each JSON file.

### Identify terminal nodes
The following script identifies the nodes that correspond to the subject and object for each snippet, and store them in a dataframe, to be used later when calculating the shortest path.

```
python identify_terminal_nodes.py 
```

### Build Corpus Graph

```
python build_corpus_graph.py
```

### Generate Node Embeddings
Be sure to note the corpus graph file tag, i.e. `corpus_graph-<tag>.pkl`

```
python generate_node_embeddings.py -tag <corpus_graph_file_tag>
```

## Relation Classification

### Build Shortest Path Vectors
To build the features for the relation classification training, run:

```
python build_shortest_path_df.py
```

#### Generate Train/Test Splits
Currently, training is not implemented with cross-validation, so this step is necessary (although it is a TODO to add cross-validation training).

```
python test_train_splits.py
```

### Train a Model on shortest path vectors for relation classification
The experimental tag needs to be provided. This is the tag attached to the split files, i.e. `X_train-<exp-tag>.pkl`

```
python train.py --model-name "dnn_wide" -itag <exp_tag>
```

### Test model on ClaimReview Claims
To test the model on the selected ClaimReview claims, run the code found in ```classify_claimreview.ipynb``` or export this code to a python script. Be sure to change the input filenames found at the top of the notebook. 

## Fact-checking
### Knowledge Stream
To fact-check the relevant claims from ClaimReview, please refer to [Knowledge Stream](https://github.com/shiralkarprashant/knowledgestream)
