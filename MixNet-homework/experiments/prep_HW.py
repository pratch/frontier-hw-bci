"""
=============================================================================
 HOMEWORK -- RUN YOUR PREPROCESSING PIPELINE  (Part 1)
=============================================================================

Reads the raw BCIC2a .mat files and writes your features to
    datasets/BCIC2a/HW_prep/2_class/subject_dependent/

Usage
    python prep_HW.py

Prerequisite
    python download_datasets.py --dataset 'BCIC2a'
"""
import mixnet.preprocessing as prep

prep.BCIC2a.HW_prep.subject_dependent_setting(
    k_folds=5, 
    pick_smp_freq=100, 
    save_path='datasets', 
    num_class=2,
    # ---------------------------------------------------------------------------
    #  Hyper-parameters of our pipeline (see HW_prep.fit_transform).
    #  Output shape: (n_trials, len(bands) * n_components, n_windows)
    #             = (n_trials, 36, 15) with the values below.
    # ---------------------------------------------------------------------------
    # same 9 sub-bands and filter order as MixNet (prep_spectral_spatial_signals.py)
    bands=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
           [24, 28], [28, 32], [32, 36], [36, 40]],
    order=5,
    n_components=4,   # 2 filters from each end of the CSP spectrum
    win_len=0.5,      # seconds -> 15 windows over the 4 s MI window
    win_step=0.25,
    norm_logvar=True,
    )

