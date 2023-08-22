#!/usr/bin/bash
for f in *.ipynb ; do jupyter nbconvert --to notebook --inplace --execute  "$f" &  done
