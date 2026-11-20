"""Publication figures (vector PDF + PNG preview) generated only from results_v2 records."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spec import DISPLAY_ID, FAULT_BY_ID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / 'results_v2'
FIG = ROOT / 'figures_v2'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6.8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'pdf.fonttype': 42, 'ps.fonttype': 42})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
OUTCOME_STYLE = [  # stack order, label, colour (validated categorical slots)
    ('correct_recovery', 'Correct recovery', '#2a78d6'),
    ('correct_abstention', 'Correct abstention', '#1baf7a'),
    ('unverified_accept', 'Unverified accept', '#6250d6'),
    ('false_rejection', 'False rejection', '#eda100'),
    ('silent_error', 'Silent error', '#e34948'),
    ('invalid_output', 'Invalid / crash', '#9a9a96'),
]
COND = [('A_unassisted', 'A: unassisted', '#2a78d6'), ('B_schema_aware', 'B: contract-guided', '#eb6834'),
        ('C_provenance_guided', 'C: provenance-guided', '#1baf7a')]
MODELS = [('claude-haiku-5-5', 'Haiku 5.5'), ('claude-sonnet-5-5', 'Sonnet 5.5'), ('claude-opus-5-5', 'Opus 5.5'),
          ('qwen2.5-coder-7b-instruct-q4_k_m', 'Qwen2.5-C-7B')]
BASE_LABEL = {'B0_naive': 'B0 naive', 'B1_schema_aware': 'B1 normalizing', 'B2_statistical': 'B2 statistical',
              'B3_contract': 'B3 contract', 'B4_manifest': 'B4 manifest', 'B5_contract_manifest': 'B5 contract+manifest',
              'B6_all_checks': 'B6 all checks', 'B7_contract_registry': 'B7 B5 + indep. anchor'}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(FIG / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def stacked(ax, labels, fractions):
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for key, lab, col in OUTCOME_STYLE:
        v = np.array([f.get(key, 0.0) for f in fractions])
        ax.barh(y, v, left=left, color=col, height=0.62, edgecolor='white', linewidth=0.8, label=lab)
        for yi, (x0, w) in enumerate(zip(left, v)):
            if key == 'silent_error' and w >= 0.03:
                ax.text(x0 + w / 2, yi, f'{100 * w:.0f}%', ha='center', va='center', color='white', fontsize=6.3,
                        fontweight='bold')
        left += v
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_baselines():
    res = pd.read_csv(R2 / 'baselines' / 'episode_results.csv', keep_default_na=False)
    main = res[res.suite == 'main']
    order = list(BASE_LABEL)
    fr = [main[main.workflow == w].outcome.value_counts(normalize=True).to_dict() for w in order]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    stacked(ax, [BASE_LABEL[w] for w in order], fr)
    ax.set_xlabel('Share of 768 real-data episodes (24 snapshots x 32 conditions)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.42, -0.2), frameon=False, handlelength=1, columnspacing=0.8)
    save(fig, 'fig_baselines_outcomes')


def fig_llm_rates():
    t7 = pd.read_csv(R2 / 'tables' / 'T7_llm_model_condition.csv')
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15), sharey=True)
    for ax, (metric, title) in zip(axes, [('silent', 'Silent-error rate\n(all 384 episodes)'),
                                          ('false_rejection', 'False-rejection rate\n(benign-change episodes)')]):
        for j, (cond, lab, col) in enumerate(COND):
            for i, (m, _) in enumerate(MODELS):
                r = t7[(t7.model == m) & (t7.condition == cond)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = i + (j - 1) * 0.22
                ax.errorbar(r[metric], yy, xerr=[[r[metric] - r[f'{metric}_lo']], [r[f'{metric}_hi'] - r[metric]]],
                            fmt='o', ms=3.6, color=col, ecolor=col, elinewidth=1.1, capsize=0,
                            label=lab if (i == 0) else None, mec='white', mew=0.5)
        ax.set_title(title, fontsize=7.3, color=INK)
        ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS])
        if ax is axes[0]:
            ax.invert_yaxis()  # shared y: invert once
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis='x', color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(left=0)
    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(1.05, -0.2), frameon=False, handletextpad=0.2,
                   columnspacing=0.8)
    save(fig, 'fig_llm_rates')


def fig_llm_faults():
    ep = pd.read_csv(R2 / 'llm' / 'episode_results_test.csv', keep_default_na=False,
                     usecols=['model', 'fid', 'condition', 'outcome'])
    ep = ep[ep.model.str.startswith('claude')]  # Claude-only, matching the pooled headline
    base = pd.read_csv(R2 / 'baselines' / 'episode_results.csv', keep_default_na=False,
                       usecols=['workflow', 'suite', 'replicate', 'fid', 'outcome'])
    base = base[(base.suite == 'main') & (base.replicate == 'A') & base.workflow.isin(['B0_naive', 'B5_contract_manifest'])]
    fids = [f for f in FAULT_BY_ID]
    cols = []
    for cond, lab, _ in COND:
        s = ep[ep.condition == cond].groupby('fid').outcome
        cols.append((lab.split(':')[0] + ' (Claude)', s.apply(lambda x: (x == 'silent_error').mean()),
                     s.apply(lambda x: (x.isin(['false_rejection', 'invalid_output'])).mean())))
    for w, lab in (('B0_naive', 'B0'), ('B5_contract_manifest', 'B5')):
        s = base[base.workflow == w].groupby('fid').outcome
        cols.append((lab, s.apply(lambda x: (x == 'silent_error').mean()),
                     s.apply(lambda x: (x == 'false_rejection').mean())))
    # rows = workflows, columns = fault conditions (wide layout for a two-column figure)
    sil = np.array([[c[1].get(f, 0) for f in fids] for c in cols])
    frj = np.array([[c[2].get(f, 0) for f in fids] for c in cols])
    fig, ax = plt.subplots(figsize=(7.1, 2.05))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('red', ['#fcfcfb', '#f6c3c2', '#e34948', '#8c1d1d'])
    im = ax.pcolormesh(np.arange(sil.shape[1] + 1) - .5, np.arange(sil.shape[0] + 1) - .5, sil, cmap=cmap, vmin=0, vmax=1,
                       edgecolors='face', linewidth=0.3)  # vector cells (was a 100-ppi raster via imshow)
    ax.set_xlim(-.5, sil.shape[1] - .5)
    ax.set_ylim(sil.shape[0] - .5, -.5)
    for i in range(len(cols)):
        for j in range(len(fids)):
            v = sil[i, j]
            if frj[i, j] >= 0.5:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='////', ec='#b07800', lw=0))
            if v >= 0.005:
                ax.text(j, i, f'{100 * v:.0f}', ha='center', va='center', fontsize=6.4,
                        color='white' if v > 0.55 else INK)
    xlab = [f"{DISPLAY_ID[f]} {FAULT_BY_ID[f].name.replace('_', ' ')}" + (' *' if FAULT_BY_ID[f].desired == 'recover' else '')
            for f in fids]
    ax.set_xticks(range(len(fids)), xlab, fontsize=6.4, rotation=60, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(cols)), [c[0] for c in cols], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(2.5, color=MUTED, lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, format=matplotlib.ticker.PercentFormatter(1.0))
    cb.set_label('Silent-error share', fontsize=6.8)
    cb.ax.tick_params(labelsize=6.4)
    cb.outline.set_visible(False)
    save(fig, 'fig_fault_heatmap')


def fig_sweep():
    t5 = pd.read_csv(R2 / 'tables' / 'T5_intensity_sweep_detection.csv', dtype={'intensity': str})
    show = ['R03', 'R04', 'R05', 'T05', 'N01', 'P04']
    wfs = [('B2_statistical', 'B2 statistical', '#eb6834', 'o'), ('B3_contract', 'B3 contract', '#1baf7a', 's'),
           ('B4_manifest', 'B4 manifest', '#2a78d6', '^')]
    short = {'R03': 'R03 nulls', 'R04': 'R04 lost records', 'R05': 'R05 unknown code', 'T05': 'T05 partition mix',
             'N01': 'N01 value change', 'P04': 'P04 forged manifest'}
    fig, axes = plt.subplots(2, 3, figsize=(3.45, 2.5), sharey=True)
    for ax, fid in zip(axes.ravel(), show):
        sub = t5[t5.fid == fid].copy()
        sub['x'] = sub.intensity.map(lambda s: 0 if s == 'one' else float(s))
        sub = sub.sort_values('x')
        xs = range(len(sub))
        for k, (w, lab, col, mk) in enumerate(wfs):
            ax.plot(np.array(list(xs)) + (k - 1) * 0.08, sub[w].to_numpy() + (k - 1) * 0.025, marker=mk, ms=3, lw=1.1, color=col, label=lab, mec='white', mew=0.4)
        ax.set_xticks(list(xs), ['1 rec.' if s == 'one' else f'{100 * float(s):g}%' for s in sub.intensity],
                      fontsize=5.6)
        ax.set_title(short[fid], fontsize=6.4)
        ax.set_ylim(-0.06, 1.06)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis='y', color=GRID, linewidth=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('Detected', fontsize=6.5)
    axes[0, 0].legend(ncol=3, loc='upper center', bbox_to_anchor=(1.75, -1.75), frameon=False, fontsize=6,
                      handlelength=1.2, columnspacing=0.6)
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    save(fig, 'fig_intensity_sweep')


def main():
    fig_baselines()
    fig_sweep()
    if (R2 / 'tables' / 'T7_llm_model_condition.csv').exists():
        fig_llm_rates()
        fig_llm_faults()
    print('figures written to', FIG)


if __name__ == '__main__':
    main()
