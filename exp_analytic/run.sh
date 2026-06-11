for k in 10 100 1000 10000; do
    for trans in "linear" "normalized" "softplus" "sigmoid"; do
        python main.py --n_m_configs 10 --n_p_configs 10 --n_eps_configs 10 --k $k --transform $trans --random_seed 42
    done
done