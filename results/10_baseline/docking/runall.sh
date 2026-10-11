#!/bin/bash
cd ./results/10_baseline/docking
while IFS=$'\t' read -r i s; do printf '%s\t%s\n' "$i" "$s"; done < tasks.tsv \
 | xargs -P 7 -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r i s <<< "{}"; conda run -n md python ./results/10_baseline/dock_one.py "$i" "$s" ./results/10_baseline/docking' \
 >> smina_scores.tsv 2>> err.log
echo DONE >> smina_scores.tsv
