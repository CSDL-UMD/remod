# Selected ClaimReview Claims

These claims were pulled from the [Google Fact Check Explorer](https://toolbox.google.com/factcheck/apis) in April 2020. Keywords from WordNet were used to search for claims with relevant relations.

They are structured in a .json format to mimic the [Augmented Google Relation Extraction Corpus](https://github.com/mjsumpter/google-relation-extraction-corpus-augmented) (A-GREC), however this mapping is not exact, and is simply for ease of machine-reading within this project. All snippets have the field `maj_vote = "yes"` as they are selected specifically for containing the designated relation. Entries that vary from the A-GREC are as follows:

* verdict_relation: Boolean indicating whether the relation contained in the snippet is factually-accurate

* verdict_claim: Boolean indicating whether the full textual claim is factually-accurate

* claim_equal_relation: Boolean indicating whether fact-checking the relation (verdict_relation) is equivalent to fact-checking the claim (verdict_claim)

* dbpedia_sub/obj: These are the terminal nodes, as found in the [FRED](http://wit.istc.cnr.it/stlab-tools/fred/) RDF graphlets. They are not always dbpedia nodes.