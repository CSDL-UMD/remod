#!/bin/bash

EXEC=snakemake

for infile in ../../../all_data/relation_extraction/data/gaivi_test/2020_02_23/rdf/*/*.rdf
do
  dir=`echo $infile | tr '/' '\n' | head -10 | tail -1`
  file=`echo $infile | tr '/' '\n' | tr '.' '\n' | head -17 | tail -1`
  timeout 15s $EXEC ../../../all_data/relation_extraction/data/rdf_processing/logs/$dir/$file.log
done