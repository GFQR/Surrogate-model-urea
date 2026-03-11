#!/bin/bash

in_file=qr_urea_dip_
out_file=qr_urea_dip_ch01_

for name in rot0.out rot45.out rot90.out rot135.out rot180.out rot225.out rot270.out rot315.out
do
#  echo $in_file$name "->" $out_file$name
  mv $in_file$name $outfile$i $out_file$name
done
