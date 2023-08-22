#!/usr/bin/bash
for f in J*/*.ipynb ; do jupyter nbconvert --to notebook --inplace --execute  "$f" &  done

