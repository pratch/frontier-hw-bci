import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score

from mixnet.utils import DataLoader
from configs import exp_config

stage2_configs = [
    {'name': 'stage2_baseline_k3_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': 'stage2_drop_06_k3', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': 'stage2_kernel_5_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 5},
    {'name': 'stage2_kernel_2_d06', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 2},
    {'name': 'stage2_kernel_2_d05', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 2},
]

screening_configs = [
    {'name': '01_baseline', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '02_lr_0005', 'lr': 0.0005, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '03_lr_0003', 'lr': 0.0003, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '04_bs_16', 'lr': 0.001, 'batch_size': 16, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '05_lr_0005_bs_16', 'lr': 0.0005, 'batch_size': 16, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '06_drop_03', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.3, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '07_drop_06', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.6, 'F1': 32, 'F2': 16, 'kernel_length': 3},
    {'name': '08_kernel_2', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 2},
    {'name': '09_kernel_5', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 32, 'F2': 16, 'kernel_length': 5},
    {'name': '10_cap_16_8', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 16, 'F2': 8, 'kernel_length': 3},
    {'name': '11_cap_48_24', 'lr': 0.001, 'batch_size': 32, 'dropout_rate': 0.5, 'F1': 48, 'F2': 24, 'kernel_length': 3},
]

def eval_saved_configs(configs, subjects, log_dir, n_folds=5):
    summary = []
    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\nEvaluating validation metrics for {cfg_name} on subjects {subjects}...")
        base_params = exp_config.get_params('HWNet', dataset='BCIC2a',
                                            train_type='subject_dependent',
                                            data_type='HW_prep', num_class=2,
                                            log_dir=f"{log_dir}/{cfg_name}")
        
        all_val_accs = []
        all_val_f1s = []
        per_subj_accs = {s: [] for s in subjects}
        per_subj_f1s = {s: [] for s in subjects}
        
        for subject in subjects:
            loader = DataLoader(subject=subject, **base_params.data_params)
            for fold in range(1, n_folds + 1):
                tf.keras.backend.clear_session()
                model_params = copy.deepcopy(base_params.model_params)
                model_params.verbose = 0
                model_params.print_summary = False
                for k, v in cfg.items():
                    if k == 'name':
                        continue
                    if k == 'lr':
                        model_params.lr = v
                        model_params.optimizer = Adam(learning_rate=v, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                    else:
                        model_params[k] = v
                
                prefix_log = f"S{subject:03d}_fold{fold:02d}"
                model = base_params.model(prefix_log=prefix_log, **model_params)
                
                weights_path = os.path.join(base_params.log_path, f"{prefix_log}_out_weights.h5")
                if not os.path.exists(weights_path):
                    print(f"Warning: {weights_path} does not exist!")
                    continue
                
                model.best_loss_weights = tf.constant([1.0])
                X_val, y_val = loader.load_val_set(fold=fold)
                _, val_eval = model.evaluate(X_val, y_val)
                
                acc = float(val_eval['test_acc'])
                f1 = float(val_eval['f1-score'])
                
                all_val_accs.append(acc)
                all_val_f1s.append(f1)
                per_subj_accs[subject].append(acc)
                per_subj_f1s[subject].append(f1)
                
        mean_acc = np.mean(all_val_accs)
        std_acc = np.std(all_val_accs)
        mean_f1 = np.mean(all_val_f1s)
        std_f1 = np.std(all_val_f1s)
        
        row = {
            'name': cfg_name,
            'mean_val_acc': mean_acc,
            'std_val_acc': std_acc,
            'mean_val_f1': mean_f1,
            'std_val_f1': std_f1,
        }
        for s in subjects:
            row[f"S{s:02d}_val_acc"] = np.mean(per_subj_accs[s])
            row[f"S{s:02d}_val_f1"] = np.mean(per_subj_f1s[s])
            
        summary.append(row)
        print(f"===> [{cfg_name}] Val Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | Val F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
    return pd.DataFrame(summary)

if __name__ == '__main__':
    print("--- Evaluating Stage 1 Screening Configs (Subjects 1, 5, 8) ---")
    df_s1 = eval_saved_configs(screening_configs, subjects=[1, 5, 8], log_dir='logs/tuning')
    df_s1.to_csv('logs/tuning/stage1_val_metrics_with_f1.csv', index=False)
    print("\nStage 1 Validation Results with F1:")
    print(df_s1[['name', 'mean_val_acc', 'std_val_acc', 'mean_val_f1', 'std_val_f1']])

    print("\n--- Evaluating Stage 2 Top Candidates (9 Subjects) ---")
    df_s2 = eval_saved_configs(stage2_configs, subjects=list(range(1, 10)), log_dir='logs/tuning')
    df_s2.to_csv('logs/tuning/stage2_val_metrics_with_f1.csv', index=False)
    print("\nStage 2 Validation Results with F1:")
    print(df_s2[['name', 'mean_val_acc', 'std_val_acc', 'mean_val_f1', 'std_val_f1']])
