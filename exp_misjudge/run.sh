for n in 100 1000 10000; do
    for p in 21 51; do
        for ds in "ml1m" "adult"; do
            for r in 0.2 0.3 0.5 0.7 0.8; do
                python main.py --dataset $ds --n_samples $n --model_ratio $r --eps_points $p --verbose
            done
        done
        python main.py --dataset synthetic --n_samples $n --eps_points $p --verbose
    done
done