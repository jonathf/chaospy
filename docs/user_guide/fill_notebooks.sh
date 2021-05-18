#!/bin/env bash

for p in $(find -name '*.ipynb'); do
  if [ "$p" != *".ipynb_checkpoints"* ]; then
    pushd $(dirname $p)
    jupyter nbconvert --to notebook --inplace \
      --ExecutePreprocessor.timeout=-1 --execute $(basename $p)
    popd
  fi
done
