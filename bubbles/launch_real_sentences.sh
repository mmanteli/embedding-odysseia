#!/bin/bash

shuffle() {
  local seed="$1"
  shift
    shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero) "$@"
}

for i in $(shuffle 42 -i 0-999 -n 205); do  # 41 was used with synthetic2
    sbatch sl-expand.sh $i
done
