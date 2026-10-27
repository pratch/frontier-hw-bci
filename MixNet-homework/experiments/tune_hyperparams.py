"""
Script to systematically tune HWNet hyperparameters on validation folds.
Stage 1: Screen configurations across representative subjects.
Stage 2: Evaluate top configurations across all 9 subjects on validation set,
         log validation accuracy and test metrics.
"""
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from configs import exp_config
from mixnet.utils import DataLoader, write_log
from tensorflow.keras.optimizers import Adam


def evaluate_config(config_dict, subjects, n_folds=5, log_dir='logs/tuning', eval_test=False):
    """
    Train and evaluate a specific hyperparameter configuration on the validation folds
    (and optionally test folds) of specified subjects.
    """
    cfg_name = config_dict['name']
    print(f"\n=======================================================")
    print(f"Evaluating Config: {cfg_name}")
    print(f"Params: {config_dict}")
    print(f"Subjects: {subjects}")
    print(f"=======================================================")
    
    # Base config
    base_params = exp_config.get_params('HWNet', dataset='BCIC2a',
                                        train_type='subject_dependent',
                                        data_type='HW_prep', num_class=2,
                                        log_dir=f"{log_dir}/{cfg_name}")
    
    all_val_accs = []
    all_val_losses = []
    all_test_accs = []
    all_test_f1s = []
    subject_val_accs = {s: [] for s in subjects}
    subject_test_accs = {s: [] for s in subjects}
    
    for subject in subjects:
        loader = DataLoader(subject=subject, **base_params.data_params)
        results = []
        
        for fold in range(1, n_folds + 1):
            tf.keras.backend.clear_session()
            
            # Prepare model params
            model_params = copy.deepcopy(base_params.model_params)
            model_params.verbose = 0
            model_params.print_summary = False
            
            # Apply hyperparameter overrides
            for k, v in config_dict.items():
                if k == 'name':
                    continue
                if k == 'lr':
                    model_params.lr = v
                    model_params.optimizer = Adam(learning_rate=v, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                else:
                    model_params[k] = v
            
            prefix_log = f"S{subject:03d}_fold{fold:02d}"
            model = base_params.model(prefix_log=prefix_log, **model_params)
            
            # Load folds
            X_train, y_train = loader.load_train_set(fold=fold)
            X_val, y_val = loader.load_val_set(fold=fold)
            
            # Train model
            model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate on validation fold
            _, val_eval = model.evaluate(X_val, y_val)
            val_acc = float(val_eval['test_acc'])
            val_loss = float(val_eval['test_loss'])
            
            all_val_accs.append(val_acc)
            all_val_losses.append(val_loss)
            subject_val_accs[subject].append(val_acc)
            
            if eval_test:
                X_test, y_test = loader.load_test_set(fold=fold)
                Y, test_eval = model.evaluate(X_test, y_test)
                test_acc = float(test_eval['test_acc'])
                test_f1 = float(test_eval['f1-score'])
                all_test_accs.append(test_acc)
                all_test_f1s.append(test_f1)
                subject_test_accs[subject].append(test_acc)
                results.append(Y)
                
                # Write log file matching standard benchmark format
                csv_file = base_params.log_path + f'/S{subject:03d}_all_results.csv'
                if fold == 1:
                    write_log(csv_file, data=test_eval.keys(), mode='w')
                write_log(csv_file, data=test_eval.values(), mode='a')
                
                print(f"[{cfg_name}] Subject {subject} Fold {fold}: Val Acc = {val_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}% (F1={test_f1:.4f})")
            else:
                print(f"[{cfg_name}] Subject {subject} Fold {fold}: Val Acc = {val_acc*100:.2f}%, Val Loss = {val_loss:.4f}")
                
        if eval_test:
            np.save(base_params.log_path + f'/S{subject:03d}_prediction_results.npy', results)
            
    mean_val_acc = np.mean(all_val_accs)
    std_val_acc = np.std(all_val_accs)
    mean_val_loss = np.mean(all_val_losses)
    per_subj_val_mean = {s: np.mean(accs) for s, accs in subject_val_accs.items()}
    
    res_dict = {
        'name': cfg_name,
        'mean_val_acc': mean_val_acc,
        'std_val_acc': std_val_acc,
        'mean_val_loss': mean_val_loss,
        **{f"S{s:02d}_val_acc": per_subj_val_mean[s] for s in subjects},
        'params': config_dict
    }
    
    if eval_test:
        mean_test_acc = np.mean(all_test_accs)
        std_test_acc = np.std(all_test_accs)
        mean_test_f1 = np.mean(all_test_f1s)
        std_test_f1 = np.std(all_test_f1s)
        per_subj_test_mean = {s: np.mean(accs) for s, accs in subject_test_accs.items()}
        res_dict.update({
            'mean_test_acc': mean_test_acc,
            'std_test_acc': std_test_acc,
            'mean_test_f1': mean_test_f1,
            'std_test_f1': std_test_f1,
            **{f"S{s:02d}_test_acc": per_subj_test_mean[s] for s in subjects}
        })
        print(f"\n---> [{cfg_name}] OVERALL: VAL ACC = {mean_val_acc*100:.2f}% | TEST ACC = {mean_test_acc*100:.2f}% ± {std_test_acc*100:.2f}% (F1 = {mean_test_f1:.4f})")
    else:
        print(f"\n---> [{cfg_name}] OVERALL VAL ACC: {mean_val_acc*100:.2f}% ± {std_val_acc*100:.2f}% | LOSS: {mean_val_loss:.4f}")
        
    return res_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='1', choices=['1', '2'],
                        help='Stage 1: screen on subset, Stage 2: run top candidates on all 9 subjects')
    parser.add_argument('--GPU', type=str, default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    
    screening_configs = [
        # Baseline
        {'name': '01_baseline', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        # Priority 1: Optimization
        {'name': '02_lr_0005', 'lr': 0.0005, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': '03_lr_0003', 'lr': 0.0003, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': '04_bs_16', 'lr': 0.001, 'batch_size': 16, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': '05_lr_0005_bs_16', 'lr': 0.0005, 'batch_size': 16, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        # Priority 2: Regularization
        {'name': '06_drop_03', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.3, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': '07_drop_06', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        # Priority 3: Kernel Length
        {'name': '08_kernel_2', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 2},
        {'name': '09_kernel_5', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 5},
        # Priority 4: Capacity
        {'name': '10_cap_16_8', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 16, 'F2': 8, 'kernel_length': 3},
        {'name': '11_cap_48_24', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 48, 'F2': 24, 'kernel_length': 3},
    ]
    
    # Top candidates selected for full 9-subject validation & test
    stage2_configs = [
        {'name': 'stage2_baseline_k3_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': 'stage2_kernel_2_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 2},
        {'name': 'stage2_drop_06_k3', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 3},
        {'name': 'stage2_kernel_2_d06', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 2},
        {'name': 'stage2_kernel_5_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 5},
    ]
    
    os.makedirs('logs/tuning', exist_ok=True)
    
    if args.stage == '1':
        print("\n=== STARTING STAGE 1: SCREENING ON SUBJECTS [1, 5, 8] ===")
        subjects = [1, 5, 8]
        results = []
        for cfg in screening_configs:
            res = evaluate_config(cfg, subjects=subjects, n_folds=5)
            results.append(res)
            df = pd.DataFrame(results)
            df.to_csv('logs/tuning/stage1_screening_results.csv', index=False)
            
        print("\n=== STAGE 1 COMPLETED ===")
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='mean_val_acc', ascending=False)
        print(df_sorted[['name', 'mean_val_acc', 'std_val_acc', 'mean_val_loss', 'S01_val_acc', 'S05_val_acc', 'S08_val_acc']])
        df_sorted.to_csv('logs/tuning/stage1_screening_results.csv', index=False)

    elif args.stage == '2':
        print("\n=== STARTING STAGE 2: FULL 9-SUBJECT EVALUATION OF TOP CANDIDATES ===")
        subjects = list(range(1, 10))
        results = []
        for cfg in stage2_configs:
            res = evaluate_config(cfg, subjects=subjects, n_folds=5, eval_test=True)
            results.append(res)
            df = pd.DataFrame(results)
            df.to_csv('logs/tuning/stage2_full_results.csv', index=False)
            
        print("\n=== STAGE 2 COMPLETED ===")
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by='mean_val_acc', ascending=False)
        cols = ['name', 'mean_val_acc', 'std_val_acc', 'mean_test_acc', 'std_test_acc', 'mean_test_f1']
        print(df_sorted[cols])
        df_sorted.to_csv('logs/tuning/stage2_full_results.csv', index=False)


if __name__ == '__main__':
    main()
