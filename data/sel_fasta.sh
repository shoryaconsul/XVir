# This Bash script extracts the FASTA sequences for the primary assemblies in FNAME
# and writes it (along with the headers) into OUTNAME. 

#!/bin/bash

FNAME="human_genome/GRCh38.p14_genomic.fna"
OUTNAME="GRCh38.fa"

rm $OUTNAME

SEL_SEQ=false
LCNT=0
PCNT=0
while read -r LINE; do
#   if [[ $LINE == \>* ]]; then
#   echo $LINE
  if [[ "$LINE" =~ ^\>.+"Homo sapiens chromosome "[0-9XY]*", GRCh38.p14 Primary Assembly" ]];
  then  # Found a matching FASTA header 
    echo $LINE
    SEL_SEQ=true
  elif [[ $LINE == \>* ]];  # FASTA header does not match
  then
    echo "Does not match"
    echo "Printed lines $PCNT of $LCNT"
    SEL_SEQ=false
  fi
  
  let LCNT=LCNT+1
  if [[ $SEL_SEQ = true ]]; then  # Print FASTA contents to output file
    let PCNT=PCNT+1
    echo $LINE >> $OUTNAME    
  fi

done < $FNAME

# : << 'COMMENT'
# COMMENT
