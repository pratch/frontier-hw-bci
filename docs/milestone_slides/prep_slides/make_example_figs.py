# Appendix figures: one subject's session T trials through each pipeline step,
# left hand vs right hand. Visualization only: CSP and the scaler are fit on all
# session T trials of this subject and no score is computed. Session E unused.
#
#   cd MixNet-homework/experiments
#   PYTHONPATH=.. python ../../docs/milestone_slides/prep_slides/make_example_figs.py [subject]
import sys

import mne
import numpy as np

from make_figs import AMBER, BANDS, FIG, GREY, OURS, TEAL, hp, plt, raw
from mixnet.utils import butter_bandpass_filter

SUBJECT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FS, MU, WIN = 100, 1, 3          # MU = index of the 8-12 Hz band, WIN = window at 1.0 s
CH = hp.CONSTANT['sel_chs']
NAMES = ('left hand', 'right hand')
COLS = (AMBER, TEAL)             # same sides as the feature map: brown = left, teal = right
t = np.arange(400) / FS


def band_labels(ax):
    ax.set_yticks([4 * b + 1.5 for b in range(len(BANDS))],
                  [f'{lo}-{hi}' for lo, hi in BANDS], fontsize=6)
    for b in range(1, len(BANDS)):
        ax.axhline(4 * b - 0.5, color='white', lw=0.8)


def typical(F, y, c):
    """Trial of class c closest to its class mean, so the example is not cherry-picked."""
    idx = np.where(y == c)[0]
    return idx[((F[idx] - F[idx].mean(0)) ** 2).sum((1, 2)).argmin()]


def fig_raw(X, trials):
    gap = 4 * np.median(X[list(trials)].std(-1))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.9), sharey=True)
    for ax, i, name, col in zip(axes, trials, NAMES, COLS):
        for k, ch in enumerate(CH):
            hl = ch in ('C3', 'C4')
            ax.plot(t, X[i, k] - k * gap, color=col if hl else GREY,
                    lw=0.8 if hl else 0.4)
        ax.set(title=f'{name} (trial {i})', xlabel='time after cue (s)', xlim=(0, 4))
        ax.set_yticks(-np.arange(len(CH)) * gap, CH, fontsize=6)
        ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/ex_raw.pdf')


def fig_filterbank(X, y, trials):
    c4 = CH.index('C4')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.8, 2.9), width_ratios=(1.7, 1))
    rows = [('raw', X[trials[0], c4])] + [
        (f'{lo}-{hi} Hz', butter_bandpass_filter(X[trials[0], c4], lo, hi, FS, 5))
        for lo, hi in BANDS]
    for k, (lab, sig) in enumerate(rows):
        # ponytail: each trace scaled to the same height, amplitudes not comparable
        ax.plot(t, sig / (4 * sig.std()) - k, color=GREY if k == 0 else AMBER,
                lw=0.8 if lab == '8-12 Hz' else 0.5)
    ax.set_yticks(-np.arange(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.set(xlabel='time after cue (s)', xlim=(0, 4))
    ax.set_title(f'C4 of the {NAMES[0]} trial, 9 bands', fontsize=8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    # contralateral check: 8-12 Hz power at C3 and C4, all trials of each class
    p = np.log(butter_bandpass_filter(X, 8, 12, FS, 5).var(-1))
    for c, (name, col) in enumerate(zip(NAMES, COLS)):
        for j, ch in enumerate(('C3', 'C4')):
            v = p[y == c, CH.index(ch)]
            bx.bar(j + (c - 0.5) * 0.36, v.mean(), 0.34, yerr=v.std() / np.sqrt(len(v)),
                   color=col, label=name if j == 0 else None)
    lo = p[:, [CH.index('C3'), CH.index('C4')]].mean() - 1
    bx.set(xticks=(0, 1), xticklabels=('C3', 'C4'), ylim=(lo, None), ylabel='mean log-variance')
    bx.set_title('8-12 Hz power, all trials', fontsize=8)
    bx.legend(fontsize=7, frameon=False)
    bx.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/ex_filterbank.pdf')


def fig_csp(X, y, S, csp, trials):
    W = csp.spatial_transform[MU][:2]                     # (2, 20), first two filters
    Xmu = butter_bandpass_filter(X, 8, 12, FS, 5)
    C = np.mean([x @ x.T / np.trace(x @ x.T) for x in Xmu], 0)
    info = mne.create_info(CH, FS, 'eeg')
    info.set_montage('standard_1020')
    fig, axes = plt.subplots(2, 3, figsize=(4.8, 2.9), width_ratios=(1, 1.6, 1.6))
    for f, w in enumerate(W):
        a = C @ w / (w @ C @ w)                           # spatial pattern of filter w
        a = a * np.sign(a[np.abs(a).argmax()])
        z = S[:, 4 * MU + f]
        hi = int(np.log(z[y == 1].var(-1)).mean() > np.log(z[y == 0].var(-1)).mean())
        mne.viz.plot_topomap(a, info, axes=axes[f, 0], show=False, contours=0,
                             cmap='RdBu_r', names=[c if c in ('C3', 'Cz', 'C4') else ''
                                                   for c in CH])
        axes[f, 0].set_title(f'filter {f + 1}: more power\nfor {NAMES[hi]}',
                             fontsize=6, color=COLS[hi])
        lim = 1.1 * np.abs(z[list(trials)]).max()
        for ax, i, name, col in zip(axes[f, 1:], trials, NAMES, COLS):
            ax.plot(t, z[i], color=col, lw=0.6)
            ax.set(ylim=(-lim, lim), xlim=(0, 4), yticks=[])
            ax.spines[['top', 'right', 'left']].set_visible(False)
            if f == 0:
                ax.set_title(f'{name} trial', fontsize=8, color=col)
            if f == 1:
                ax.set_xlabel('time after cue (s)')
            else:
                ax.set_xticklabels([])
    fig.tight_layout()
    fig.savefig(f'{FIG}/ex_csp.pdf')


def fig_features(F, trials):
    lim = np.abs(F[list(trials)]).max()
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.9), sharey=True)
    for ax, i, name, col in zip(axes, trials, NAMES, COLS):
        im = ax.imshow(F[i], aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                       extent=(0.125, 3.875, F.shape[1] - 0.5, -0.5),
                       interpolation='nearest')
        band_labels(ax)
        ax.set(xlim=(0, 4), xlabel='window centre (s)')
        ax.set_title(f'{name} trial', color=col)
    axes[0].set_ylabel('band (Hz), 4 CSP filters each')
    fig.colorbar(im, ax=axes, pad=0.02).set_label('standardized log-variance', fontsize=8)
    fig.savefig(f'{FIG}/ex_features.pdf', bbox_inches='tight')


def fig_separation(F, y):
    tw = 0.25 + 0.25 * np.arange(F.shape[2])
    fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.3), width_ratios=(1, 1, 1.1))
    for f, ax in enumerate(axes[:2]):
        for c, (name, col) in enumerate(zip(NAMES, COLS)):
            v = F[y == c, 4 * MU + f]
            m, se = v.mean(0), v.std(0) / np.sqrt(len(v))
            ax.fill_between(tw, m - se, m + se, color=col, alpha=0.2, lw=0)
            ax.plot(tw, m, color=col, label=name)
        ax.set(title=f'8-12 Hz, filter {f + 1}', xlabel='window centre (s)', xlim=(0, 4),
               xticks=(0, 2, 4))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('class mean (training SD)')
    axes[0].legend(fontsize=7, frameon=False, loc='center left', bbox_to_anchor=(0.3, 0.55))
    ax = axes[2]
    for c, (name, col) in enumerate(zip(NAMES, COLS)):
        ax.scatter(F[y == c, 4 * MU, WIN], F[y == c, 4 * MU + 1, WIN], s=8,
                   color=col, alpha=0.7, lw=0, label=name)
    ax.set(title='one dot per trial\n(1.0 s window)',
           xlabel='8-12 Hz filter 1', ylabel='8-12 Hz filter 2')
    ax.title.set_fontsize(8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FIG}/ex_separation.pdf')


if __name__ == '__main__':
    ids = raw.chanel_selection(CH)
    X, y, _, _ = hp._load_bcic2a(SUBJECT, FS, 2, ids)   # session E discarded
    F, st = hp.fit_transform(X, y, FS, **OURS)
    S = st['csp'].transform(X)
    trials = (typical(F, y, 0), typical(F, y, 1))
    fig_raw(X, trials)
    fig_filterbank(X, y, trials)
    fig_csp(X, y, S, st['csp'], trials)
    fig_features(F, trials)
    fig_separation(F, y)
    print(f'S{SUBJECT}: example trials {trials}, shapes X {X.shape} S {S.shape} F {F.shape}')
