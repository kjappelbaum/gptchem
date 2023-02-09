for seed in 0 1 2 3 4 5 6 7 8 9; do
    for trainsize in 10 20 50 100 200; do
        python3 run_crabnet.py $trainsize $seed
    done
done
