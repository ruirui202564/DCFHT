## DCFHT: online learning from Drifting Capricious data streams with Flexible Hoeffding Tree

## Overview
This repository contains datasets and implementation codes of the paper, titled "Online Learning from Drifting Capricious Data Streams with Flexible Hoeffding Tree".

## Requirements:
  numpy
  pandas
  openpyxl
  os
  skmultiflow (We have modified the original codes provided in https://scikit-multiflow.github.io/. And the modified library is provided.)
  python 3.7

## Parameters
drift threshold b: it can be adjusted in Line 256 in DCFHT.py.
disappearance threshold s_{max}: it can be adjusted in Line 80 in findAttr.py.

## Run Example
In DCFHT folder, we provide two files for different running examples:
  1."main.py" is used to run datasets without discrete features;
  2."main_discrete.py" is used to run datasets with discrete features.
  The tested dataset can be changed at the beginning of the files. They can be run without any processing.
