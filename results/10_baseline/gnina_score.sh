#!/bin/bash
# gnina GPU 재점수 — 배치 50, 실패 시 25 → 10 으로 축소
set -u
D=./results/10_baseline/docking
G=<path-to>/gnina
P=./data/cdk2_clean.pdb
OUT=./results/10_baseline/gnina_scores.tsv
: > $OUT
run_batch () {  # $1=파일목록 $2=배치크기
  local files=($1); local bs=$2; local ok=1
  for ((k=0;k<${#files[@]};k+=bs)); do
    local chunk=("${files[@]:k:bs}")
    local tmp=$(mktemp /tmp/gb_XXXX.sdf)
    local ids=()
    for f in "${chunk[@]}"; do cat "$D/$f" >> $tmp; ids+=("$(basename $f .sdf | sed 's/docked_//')"); done
    local res=$($G -r $P -l $tmp --score_only --cnn_scoring rescore --device 0 2>/dev/null \
      | grep -E "^(Affinity|CNNscore|CNNaffinity):" | awk '{print $2}')
    rm -f $tmp
    local n=$(echo "$res" | grep -c .)
    if [ "$n" -ne $((${#chunk[@]}*3)) ]; then echo "  배치 실패 (기대 $((${#chunk[@]}*3)) 실제 $n)" >&2; ok=0; break; fi
    local j=0
    for id in "${ids[@]}"; do
      local a=$(echo "$res" | sed -n "$((j*3+1))p"); local cs=$(echo "$res" | sed -n "$((j*3+2))p"); local ca=$(echo "$res" | sed -n "$((j*3+3))p")
      printf "%s\t%s\t%s\t%s\n" "$id" "$a" "$cs" "$ca" >> $OUT
      j=$((j+1))
    done
    printf "\r  gnina %d/%d" $((k+${#chunk[@]})) ${#files[@]} >&2
  done
  return $((1-ok))
}
FILES=$(cd $D && ls docked_*.sdf | sort -t_ -k2 -n)
for BS in 50 25 10; do
  : > $OUT
  if run_batch "$FILES" $BS; then echo; echo "  ★배치 $BS 성공: $(wc -l < $OUT)개"; exit 0; fi
  echo "  → 배치 $BS 실패, 축소 재시도" >&2
done
echo "  전부 실패" >&2; exit 1
