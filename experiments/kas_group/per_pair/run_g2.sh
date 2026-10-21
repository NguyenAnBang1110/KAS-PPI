for v in kg seek; do for p in source evidence; do for f in 1 2 3 4 5 6 7 8 9 10; do
python3 kas_group_x.py $v $p $f 500 > g10/log_kasgx_${v}_${p}_f$f.txt 2>&1; done; done; done
touch g10/G2_DONE
