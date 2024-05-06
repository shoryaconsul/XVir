#!/bin/bash

# Extract headers and sequences/reds from FASTQ file to form reads.

while getopts f:o: FLAG
do
  case "${FLAG}" in
    f) FASTQ=${OPTARG};;  # Path to FASTQ file
    o) OUTPATH=${OPTARG};;  # Output file (full path or relative path)
    *) echo "$(basename $0): Invalid command line option: -$FLAG" ;;
  esac
done


TMP="mixture/tmp.fa"
cat $FASTQ | awk '{if (NR%4==1) {print ">"substr($0,2)} else if (NR%4==2) {print $0}}' > $TMP
# cat $TMP | awk '{if (NR%2==0) {if ($0 ~ "^[ACGT][ACGT]+[ACGT]$") {print $0}}}' > "mixture/del.fa"

# Filter out reads that contain just A,C,G,T (a bug in ART Illumina read generation)
cat $TMP | awk '{if (NR%2==1) {header=$0} else if($0 ~ "^[ACGT][ACGT]+[ACGT]$") {print header"\n"$0}}' > $OUTPATH 
rm $TMP

#cat $FASTQ | awk '{if (NR%4==1) {print ">"substr($0,2)} else if (NR%4==2) {print $0}}' > $OUTPATH