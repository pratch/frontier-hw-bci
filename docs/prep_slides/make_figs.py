# Figures for the preprocessing slides. Uses session T only; session E is
# loaded by the shared loader but never enters any number or plot.
#
#   cd MixNet-homework/experiments
#   PYTHONPATH=.. python ../../docs/prep_slides/make_figs.py
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

from mixnet.preprocessing.BCIC2a import HW_prep as hp, raw

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
BANDS = [[4 + 4 * i, 8 + 4 * i] for i in range(9)]
HP = dict(bands=BANDS, n_components=4, order=5, norm_logvar=True)
OURS = dict(win_len=0.5, win_step=0.25, **HP)
CLASSIC = dict(win_len=4.0, win_step=4.0, **HP)
TEAL, AMBER, GREY = '#1B7F79', '#9A6A00', '#888888'

# same font as the slides (Fira Sans from TeX Live) if it can be found
try:
    from matplotlib import font_manager
    import subprocess
    _fira = subprocess.run(['kpsewhich', 'FiraSans-Book.otf'], capture_output=True,
                           text=True).stdout.strip()
    font_manager.fontManager.addfont(_fira)
    plt.rcParams['font.family'] = 'Fira Sans'
except Exception:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'axes.grid': False, 'font.size': 9})


def lda():
    return LDA(solver='lsqr', shrinkage='auto')


def run(subject):
    ids = raw.chanel_selection(hp.CONSTANT['sel_chs'])
    X, y, _, _ = hp._load_bcic2a(subject, 100, 2, ids)   # session E discarded
    per_win, all_win, classic = [], [], []
    # same folds as the real pipeline; score on the validation fold only
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        F, st = hp.fit_transform(X[tr], y[tr], 100, **OURS)
        Fv = hp.transform(X[va], st, 100)
        per_win.append([lda().fit(F[:, :, w], y[tr]).score(Fv[:, :, w], y[va])
                        for w in range(F.shape[2])])
        all_win.append(lda().fit(F.reshape(len(F), -1), y[tr])
                       .score(Fv.reshape(len(Fv), -1), y[va]))
        C, st = hp.fit_transform(X[tr], y[tr], 100, **CLASSIC)
        Cv = hp.transform(X[va], st, 100)
        classic.append(lda().fit(C[:, :, 0], y[tr]).score(Cv[:, :, 0], y[va]))
    # class contrast over all of session T, in units of the training SD
    F, _ = hp.fit_transform(X, y, 100, **OURS)
    diff = F[y == 1].mean(0) - F[y == 0].mean(0)
    return np.mean(per_win, 0), np.mean(all_win), np.mean(classic), diff


def plot_acc(per_win, all_win, classic):
    t = 0.25 + 0.25 * np.arange(per_win.shape[1])   # window centre, s after cue
    m, se = per_win.mean(0) * 100, per_win.std(0, ddof=1) / 3 * 100
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.fill_between(t, m - se, m + se, color=TEAL, alpha=0.18, lw=0)
    ax.plot(t, m, 'o-', color=TEAL, ms=3.5)
    a, c = all_win.mean() * 100, classic.mean() * 100
    ax.axhline(a, color=TEAL, ls='--')
    ax.axhline(c, color=AMBER, ls='--')
    ax.axhline(50, color=GREY, ls=':', lw=1)
    ax.text(3.95, a + 0.6, f'all 15 windows together: {a:.1f}%', color=TEAL,
            ha='right', va='bottom', fontsize=8)
    ax.text(3.95, c - 0.6, f'one 4 s window (classic CSP): {c:.1f}%', color=AMBER,
            ha='right', va='top', fontsize=8)
    ax.text(2.3, 72, 'one 0.5 s window only', color=TEAL, fontsize=8)
    ax.text(3.95, 50.6, 'chance', color=GREY, ha='right', va='bottom', fontsize=7)
    ax.set(xlabel='window centre (s after cue)', ylabel='LDA val accuracy (%)',
           xlim=(0, 4), ylim=(45, 92))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/acc_by_window.pdf')


def plot_map(diff):
    d = diff.mean(0)
    lim = np.abs(d).max()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.imshow(d, aspect='auto', cmap='BrBG', vmin=-lim, vmax=lim,
                   extent=(0.125, 3.875, d.shape[0], 0), interpolation='nearest')
    for b in range(1, len(BANDS)):
        ax.axhline(4 * b, color='white', lw=0.8)
    ax.set_yticks([4 * b + 2 for b in range(len(BANDS))],
                  [f'{lo}-{hi}' for lo, hi in BANDS], fontsize=7)
    ax.set(xlim=(0, 4), xlabel='window centre (s after cue)', ylabel='band (Hz), 4 CSP comps each')
    fig.colorbar(im, ax=ax, pad=0.02).set_label('right - left (training SD)', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/feature_map.pdf')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    with ProcessPoolExecutor(9) as ex:
        res = list(ex.map(run, range(1, 10)))
    per_win, all_win, classic, diff = (np.array(r) for r in zip(*res))
    plot_acc(per_win, all_win, classic)
    plot_map(diff)
    m = per_win.mean(0) * 100
    print('subject | all 15 windows | classic')
    for s in range(9):
        print(f'S{s + 1} | {all_win[s] * 100:.1f} | {classic[s] * 100:.1f}')
    print(f'mean | {all_win.mean() * 100:.1f} | {classic.mean() * 100:.1f}')
    print('per-window mean acc:', np.round(m, 1))
    print(f'best window centre {0.25 + 0.25 * m.argmax():.2f} s ({m.max():.1f}%), '
          f'first {m[0]:.1f}%, last {m[-1]:.1f}%')
