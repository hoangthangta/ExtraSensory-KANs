#--------------------------------------------------------------------------
# ExtraSensory: 3 seeds × 5 folds
# Seeds: 0, 1, 2
# Folds: 0, 1, 2, 3, 4
#--------------------------------------------------------------------------

for seed in 0 1 2; do

    # Multilabel — 51 labels
    # MLP — 71925 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "mlp" --fold "$f" --hidden_layers "256" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # SechKAN — 71947 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "sech_kan" --fold "$f" --hidden_layers "256" --num_grids 4 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --norm1_type "layer" --norm2_type "" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # EfficientKAN — 71760 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "efficient_kan" --fold "$f" --hidden_layers "26" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # FastKAN — 72624 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "fast_kan" --fold "$f" --hidden_layers "29" --num_grids 8 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # BSRBF-KAN — 72544 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "bsrbf_kan" --fold "$f" --hidden_layers "29" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --norm_type "layer" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # TabM — 71002 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "multilabel" --model "tab_m" --fold "$f" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # Main activity — 6 classes
    # MLP — 60360 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "mlp" --fold "$f" --hidden_layers "256" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # SechKAN — 60382 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "sech_kan" --fold "$f" --hidden_layers "256" --num_grids 4 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --norm1_type "layer" --norm2_type "" --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # EfficientKAN — 60060 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "efficient_kan" --fold "$f" --hidden_layers "26" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # FastKAN — 60834 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "fast_kan" --fold "$f" --hidden_layers "29" --num_grids 8 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # BSRBF-KAN — 60799 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "bsrbf_kan" --fold "$f" --hidden_layers "29" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --norm_type "layer" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

    # TabM — 59648 params
    for f in 0 1 2 3 4; do
        python run_extra_sen.py --task "main" --model "tab_m" --fold "$f" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed "$seed" --note "seed_${seed}_fold_${f}"
        sleep 5
    done

done
