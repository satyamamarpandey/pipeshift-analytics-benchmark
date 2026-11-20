"""Compute every reported table and number from episode-level records.

Outputs: results_v2/tables/*.csv, results_v2/summary.json and paper/generated_macros.tex.
Every number used in the manuscript is emitted as a LaTeX macro from this script.
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from spec import DISPLAY_ID, FAULTS
from stats import bootstrap_diff, cluster_bootstrap_mean, holm, permutation_test, wilson

ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / 'results_v2'
TAB = R2 / 'tables'
OUTCOMES = ['correct_recovery', 'correct_abstention', 'false_rejection', 'silent_error',
            'unverified_accept', 'invalid_output']
BASE_ORDER = ['B0_naive', 'B1_schema_aware', 'B2_statistical', 'B3_contract', 'B4_manifest',
              'B5_contract_manifest', 'B6_all_checks']
COND_ORDER = ['A_unassisted', 'B_schema_aware', 'C_provenance_guided']
IMPACT = {f.fid for f in FAULTS if f.metric_impact}

summary: dict = {}
macros: dict[str, str] = {}


def mac(name: str, value) -> None:
    assert re.fullmatch(r'[A-Za-z]+', name), name
    assert name not in macros, name
    macros[name] = str(value)


def pct(x: float, d: int = 1) -> str:
    return f'{100 * x:.{d}f}'


def outcome_counts(df: pd.DataFrame) -> dict:
    vc = df.outcome.value_counts()
    return {o: int(vc.get(o, 0)) for o in OUTCOMES}


# --------------------------------------------------------------------------- data
def data_tables() -> None:
    prof = pd.read_csv(R2 / 'data_profile' / 'natural_quality_profile.csv')
    samp = pd.read_csv(R2 / 'data_profile' / 'snapshot_samples.csv')
    study = prof[prof.month.str.startswith('2025')]
    keep = ['month', 'raw_rows', 'admissible_distinct_rows', 'pickup_outside_month', 'dropoff_before_pickup',
            'distance_nonpositive', 'distance_over_100mi', 'fare_negative', 'fare_over_1000',
            'payment_code_0_flex', 'required_field_null', 'exact_duplicate_rows', 'has_cbd_congestion_fee']
    prof[keep].to_csv(TAB / 'T1_natural_quality_profile.csv', index=False)
    raw_total = int(prof.raw_rows.sum())
    study_raw = int(study.raw_rows.sum())
    excl = 1 - study.admissible_distinct_rows.sum() / study_raw
    snaps = samp[samp.part.str.endswith('snapshot') & samp.month.str.startswith('2025')]
    summary['data'] = {'months_downloaded': len(prof), 'raw_rows_all_files': raw_total,
                       'study_months': len(study), 'study_raw_rows': study_raw,
                       'study_admissible_rows': int(study.admissible_distinct_rows.sum()),
                       'excluded_fraction': excl, 'snapshots': len(snaps),
                       'snapshot_rows': int(snaps.rows.iloc[0]),
                       'records_in_snapshots': int(snaps.rows.sum()),
                       'fare_negative_study': int(study.fare_negative.sum()),
                       'distance_nonpositive_study': int(study.distance_nonpositive.sum()),
                       'outside_month_study': int(study.pickup_outside_month.sum()),
                       'flex_fare_study': int(study.payment_code_0_flex.sum()),
                       'cbd_fee_added_month': str(prof[prof.has_cbd_congestion_fee].month.min()),
                       'card_share_range': [float(snaps.card_share.min()), float(snaps.card_share.max())]}
    d = summary['data']
    mac('NMonthsDownloaded', d['months_downloaded'])
    mac('RawRowsAll', f"{raw_total / 1e6:.1f}")
    mac('StudyRawRows', f"{study_raw / 1e6:.1f}")
    mac('StudyAdmissibleRows', f"{d['study_admissible_rows'] / 1e6:.1f}")
    mac('ExcludedPct', pct(excl))
    mac('NSnapshots', d['snapshots'])
    mac('SnapshotRows', f"{d['snapshot_rows']:,}".replace(',', '{,}'))
    mac('RecordsInSnapshots', f"{d['records_in_snapshots']:,}".replace(',', '{,}'))
    mac('FareNegativeStudy', f"{d['fare_negative_study']:,}".replace(',', '{,}'))
    mac('OutsideMonthStudy', d['outside_month_study'])
    mac('FlexFareStudyPct', pct(d['flex_fare_study'] / study_raw))
    mac('CardShareMin', pct(d['card_share_range'][0]))
    mac('CardShareMax', pct(d['card_share_range'][1]))


# --------------------------------------------------------------------- baselines
def _bad_delta(main: pd.DataFrame, wf: str) -> dict:
    bad = ['silent_error', 'false_rejection', 'unverified_accept']
    cnt = lambda w: main[(main.workflow == w) & main.outcome.isin(bad)].groupby('fid').size()
    return cnt(wf).sub(cnt('B6_all_checks'), fill_value=0).astype(int).to_dict()



def baseline_tables() -> pd.DataFrame:
    res = pd.read_csv(R2 / 'baselines' / 'episode_results.csv', keep_default_na=False, dtype={'intensity': str})
    main = res[res.suite == 'main']
    rows = []
    for wf in BASE_ORDER + ['B7_contract_registry']:
        x = main[main.workflow == wf]
        c = outcome_counts(x)
        n = len(x)
        rec = x[x.desired == 'recover']
        abst = x[x.desired == 'abstain']
        k_sil = c['silent_error']
        lo, hi = wilson(k_sil, n)
        fr = int((rec.outcome == 'false_rejection').sum())
        frl, frh = wilson(fr, len(rec))
        des = c['correct_recovery'] + c['correct_abstention']
        imp = x[x.fid.isin(IMPACT)]
        rows.append({'workflow': wf, 'episodes': n, **c,
                     'silent_rate': k_sil / n, 'silent_lo': lo, 'silent_hi': hi,
                     'silent_rate_impact_faults': (imp.outcome == 'silent_error').mean(),
                     'false_rejection_rate': fr / len(rec), 'fr_lo': frl, 'fr_hi': frh,
                     'desired_rate': des / n,
                     'abstain_detection_rate': (abst.outcome == 'correct_abstention').mean(),
                     'median_ms': 1000 * x.seconds.median()})
    t2 = pd.DataFrame(rows)
    t2.to_csv(TAB / 'T2_baselines_main.csv', index=False)
    summary['baselines_main'] = t2.set_index('workflow').to_dict('index')
    names = {'B0_naive': 'Bzero', 'B1_schema_aware': 'Bone', 'B2_statistical': 'Btwo', 'B3_contract': 'Bthree',
             'B4_manifest': 'Bfour', 'B5_contract_manifest': 'Bfive', 'B6_all_checks': 'Bsix',
             'B7_contract_registry': 'Bseven'}
    for r in rows:
        k = names[r['workflow']]
        mac(f'Sil{k}', r['silent_error'])
        mac(f'SilPct{k}', pct(r['silent_rate']))
        mac(f'FR{k}', r['false_rejection'])
        mac(f'FRPct{k}', pct(r['false_rejection_rate']))
        mac(f'Des{k}', pct(r['desired_rate']))
        mac(f'Unv{k}', r['unverified_accept'])
    mac('NMainEpisodes', int(rows[0]['episodes']))
    mac('NFaults', len(FAULTS))
    mac('NImpactFaults', len(IMPACT))
    mac('NRecoverFaults', sum(f.desired == 'recover' for f in FAULTS))
    mac('NAbstainFaults', sum(f.desired == 'abstain' for f in FAULTS))

    # per-family silent errors
    fam = main[main.workflow.isin(BASE_ORDER)].groupby(['family', 'workflow']).outcome.apply(
        lambda s: (s == 'silent_error').mean()).unstack()[BASE_ORDER]
    fam.to_csv(TAB / 'T3_baselines_family_silent_rate.csv')
    # per-fault outcome matrix (modal outcome and count over 24 snapshots)
    pf = main[main.workflow.isin(BASE_ORDER + ['B7_contract_registry'])].groupby(['fid', 'workflow']).outcome.agg(
        lambda s: f"{s.value_counts().index[0]}:{s.value_counts().iloc[0]}/{len(s)}").unstack()
    pf.to_csv(TAB / 'T3b_baselines_per_fault.csv')

    # ablations
    ab_rows = []
    ref = outcome_counts(main[main.workflow == 'B6_all_checks'])
    for wf in sorted(w for w in main.workflow.unique() if w.startswith('ABL_')):
        c = outcome_counts(main[main.workflow == wf])
        x = main[main.workflow == wf]
        ab_rows.append({'removed_component': wf.replace('ABL_minus_', ''), **c,
                        'delta_silent': c['silent_error'] - ref['silent_error'],
                        'delta_false_rejection': c['false_rejection'] - ref['false_rejection'],
                        'delta_unverified': c['unverified_accept'] - ref['unverified_accept'],
                        'faults_affected': '|'.join(
                            f'{DISPLAY_ID[fid]}(+{d})' for fid, d in _bad_delta(main, wf).items() if d > 0)})
    ab = pd.DataFrame(ab_rows).sort_values('delta_silent', ascending=False)
    ab.to_csv(TAB / 'T4_ablation.csv', index=False)
    summary['ablation'] = ab.to_dict('records')
    mac('AblHashDeltaSil', int(ab.set_index('removed_component').loc['hash_check', 'delta_silent']))
    mac('AblHashDeltaUnv', int(ab.set_index('removed_component').loc['hash_check', 'delta_unverified']))
    mac('AblDedupDeltaSil', int(ab.set_index('removed_component').loc['dedup_exact', 'delta_silent']))
    mac('AblStatsRemovedFR', -int(ab.set_index('removed_component').loc['stats_check', 'delta_false_rejection']))
    mac('AblStatsDeltaFR', int(ab.set_index('removed_component').loc['stats_check', 'delta_false_rejection']))
    mac('AblStatsDeltaSil', int(ab.set_index('removed_component').loc['stats_check', 'delta_silent']))

    # intensity sweep: detection rate (correct_abstention) per fault, level, workflow
    sw = res[res.suite == 'sweep']
    t5 = sw.groupby(['fid', 'intensity', 'workflow']).outcome.apply(
        lambda s: (s == 'correct_abstention').mean()).unstack()
    t5 = t5[[w for w in BASE_ORDER + ['B7_contract_registry'] if w in t5.columns]]
    t5.to_csv(TAB / 'T5_intensity_sweep_detection.csv')
    summary['sweep_stats_detection_one_record'] = {
        fid: float(t5.loc[(fid, 'one'), 'B2_statistical']) for fid in t5.index.get_level_values(0).unique()}

    # trust anchor experiment
    ta = res[(res.fid.isin(['F00', 'N01', 'P04'])) &
             res.workflow.isin(['B5_contract_manifest', 'B7_contract_registry'])]
    t6 = ta.groupby(['fid', 'suite', 'intensity', 'workflow']).outcome.agg(
        lambda s: f"{(s == 'correct_abstention').sum()}/{len(s)} abstain; {(s == 'silent_error').sum()} silent; "
                  f"{(s == 'correct_recovery').sum()} recovered").unstack()
    t6.to_csv(TAB / 'T6_trust_anchor.csv')
    p4 = res[(res.fid == 'P04')]
    for wf, k in (('B5_contract_manifest', 'Bfive'), ('B7_contract_registry', 'Bseven')):
        x = p4[p4.workflow == wf]
        mac(f'TamperSil{k}', int((x.outcome == 'silent_error').sum()))
        mac(f'TamperN{k}', len(x))
    n04 = main[(main.fid == 'N04') & (main.workflow == 'B5_contract_manifest')]
    mac('KmSilBfive', int((n04.outcome == 'silent_error').sum()))
    dec = main[(main.workflow == 'B2_statistical') & (main.desired == 'recover') & (main.outcome == 'false_rejection')]
    summary['stats_false_rejections_by_snapshot'] = dec.groupby('snapshot').size().to_dict()
    mac('StatsFRDistShift', int((dec.fid == 'D01').sum()))
    mac('StatsFRNatural', int((dec.fid != 'D01').sum()))
    return main


# --------------------------------------------------------------------------- LLM
def llm_tables(base_main: pd.DataFrame) -> None:
    path = R2 / 'llm' / 'episode_results_test.csv'
    if not path.exists():
        summary['llm'] = 'NOT RUN'
        return
    ep = pd.read_csv(path, keep_default_na=False)
    gens = pd.concat([pd.read_csv(f) for f in sorted((R2 / 'llm' / 'logs').glob('generations_test_*.csv'))],
                     ignore_index=True)
    assert ep.groupby('gen_id').size().eq(384).all(), 'every generation must have 384 episodes'
    assert set(ep.gen_id) == set(gens.gen_id), 'every logged generation must be evaluated'
    # Valid detection (post hoc, after independent review): an abstention counts only if the same generation
    # recovered the clean control (F00) on the same snapshot, i.e. the code is not simply failing on everything.
    f00 = ep[ep.fid == 'F00'].assign(f00ok=lambda d: d.outcome == 'correct_recovery')[['gen_id', 'snapshot', 'f00ok']]
    ep = ep.merge(f00, on=['gen_id', 'snapshot'], how='left', validate='many_to_one')
    ep['valid_abstention'] = (ep.outcome == 'correct_abstention') & ep.f00ok
    per_gen = ep.groupby(['model', 'condition', 'gen_id']).apply(lambda g: pd.Series({
        'valid_detection': g[g.desired == 'abstain'].valid_abstention.mean(),
        'silent': (g.outcome == 'silent_error').mean(),
        'silent_impact': (g[g.fid.isin(IMPACT)].outcome == 'silent_error').mean(),
        'false_rejection': (g[g.desired == 'recover'].outcome == 'false_rejection').mean(),
        'desired': g.outcome.isin(['correct_recovery', 'correct_abstention']).mean(),
        'unverified': (g.outcome == 'unverified_accept').mean(),
        'invalid': (g.outcome == 'invalid_output').mean(),
        'detection': (g[g.desired == 'abstain'].outcome == 'correct_abstention').mean(),
        'recovery': (g[g.desired == 'recover'].outcome == 'correct_recovery').mean(),
        # Post-hoc descriptors (docs/DEVIATIONS.md): a constant rejector scores 0.5 balanced accuracy.
        'balanced_acc': 0.5 * ((g[g.desired == 'recover'].outcome == 'correct_recovery').mean()
                               + (g[g.desired == 'abstain'].outcome == 'correct_abstention').mean()),
        'functional': float((g[g.fid == 'F00'].outcome == 'correct_recovery').any()),
        'explicit_abstain_share': (g.signal == 'abstain').sum() / max(1, g.signal.isin(['abstain', 'exception']).sum()),
        'median_ms': 1000 * g.seconds.median()}), include_groups=False).reset_index()
    per_gen = per_gen.merge(gens[['gen_id', 'output_tokens', 'cost_usd', 'code_lines', 'code_extracted']],
                            on='gen_id', how='left')
    per_gen.to_csv(TAB / 'T7a_llm_per_generation.csv', index=False)
    rows = []
    for (model, cond), g in per_gen.groupby(['model', 'condition']):
        r = {'model': model, 'condition': cond, 'n_generations': len(g), 'episodes': len(g) * 384}
        for k in ('silent', 'silent_impact', 'false_rejection', 'desired', 'unverified', 'invalid', 'detection',
                  'recovery', 'balanced_acc', 'explicit_abstain_share', 'valid_detection'):
            m, lo, hi = cluster_bootstrap_mean(g[k])
            r.update({k: m, f'{k}_lo': lo, f'{k}_hi': hi})
        r.update({'mean_output_tokens': g.output_tokens.mean(), 'total_cost_usd': g.cost_usd.sum(),
                  'mean_code_lines': g.code_lines.mean(), 'gens_without_code': int((~g.code_extracted).sum()),
                  'functional_generations': int(g.functional.sum())})
        rows.append(r)
    t7 = pd.DataFrame(rows)
    t7['condition'] = pd.Categorical(t7.condition, COND_ORDER)
    t7 = t7.sort_values(['model', 'condition'])
    t7.to_csv(TAB / 'T7_llm_model_condition.csv', index=False)
    summary['llm_model_condition'] = t7.astype({'condition': str}).to_dict('records')
    summary['llm_total_cost_usd'] = float(gens.cost_usd.sum())
    summary['llm_generations'] = int(len(gens))
    summary['llm_episode_executions'] = int(len(ep))
    mac('NGenerations', len(gens))
    mac('NLLMExecutions', f"{len(ep):,}".replace(',', '{,}'))
    mac('LLMCost', f"{gens.cost_usd.sum():.2f}")
    allg = pd.concat([pd.read_csv(f) for f in (R2 / 'llm' / 'logs').glob('generations_*.csv')])
    mac('LLMCostAll', f"{allg.cost_usd.sum():.2f}")
    summary['llm_total_cost_incl_pilot_usd'] = float(allg.cost_usd.sum())
    mac('NModels', gens.model.nunique())
    mtag = {'claude-haiku-5-5': 'Haiku', 'claude-sonnet-5-5': 'Sonnet', 'claude-opus-5-5': 'Opus',
            'qwen2.5-coder-7b-instruct-q4_k_m': 'Qwen'}
    ctag = {'A_unassisted': 'A', 'B_schema_aware': 'B', 'C_provenance_guided': 'C'}
    for r in rows:
        if r['model'] not in mtag:
            continue
        k = mtag[r['model']] + ctag[r['condition']]
        mac(f'Sil{k}', pct(r['silent']))
        mac(f'SilLo{k}', pct(r['silent_lo']))
        mac(f'SilHi{k}', pct(r['silent_hi']))
        mac(f'FR{k}', pct(r['false_rejection']))
        mac(f'Des{k}', pct(r['desired']))
        mac(f'Inv{k}', pct(r['invalid']))
        mac(f'Det{k}', pct(r['detection']))
        mac(f'Rec{k}', pct(r['recovery']))
        mac(f'Bal{k}', pct(r['balanced_acc']))
        mac(f'Func{k}', r['functional_generations'])
        mac(f'Expl{k}', pct(r['explicit_abstain_share'], 0))
    # pooled over models per condition (generation = unit)
    pooled = []
    for cond, g in per_gen.groupby('condition'):
        m, lo, hi = cluster_bootstrap_mean(g.silent)
        fm, flo, fhi = cluster_bootstrap_mean(g.false_rejection)
        pooled.append({'condition': cond, 'n_generations': len(g), 'silent': m, 'silent_lo': lo, 'silent_hi': hi,
                       'false_rejection': fm, 'fr_lo': flo, 'fr_hi': fhi})
        mac(f'PooledSil{ctag[cond]}', pct(m))
        mac(f'PooledSilLo{ctag[cond]}', pct(lo))
        mac(f'PooledSilHi{ctag[cond]}', pct(hi))
    pd.DataFrame(pooled).to_csv(TAB / 'T7b_llm_pooled_condition.csv', index=False)
    # Claude-only pooling (all Claude generations were functional on the clean control in most cells;
    # the open-weight model is reported separately because most of its guided generations never ran).
    cl = per_gen[per_gen.model.str.startswith('claude')]
    cl_rows = []
    for cond, g in cl.groupby('condition'):
        row = {'condition': cond, 'n_generations': len(g)}
        for k, tagk in (('silent', 'Sil'), ('false_rejection', 'FR'), ('balanced_acc', 'Bal'),
                        ('silent_impact', 'SilImp'), ('valid_detection', 'VDet')):
            m, lo, hi = cluster_bootstrap_mean(g[k])
            row.update({k: m, f'{k}_lo': lo, f'{k}_hi': hi})
            mac(f'Claude{tagk}{ctag[cond]}', pct(m))
            mac(f'Claude{tagk}Lo{ctag[cond]}', pct(lo))
            mac(f'Claude{tagk}Hi{ctag[cond]}', pct(hi))
        cl_rows.append(row)
    pd.DataFrame(cl_rows).to_csv(TAB / 'T7c_llm_claude_pooled_condition.csv', index=False)
    summary['llm_claude_pooled'] = cl_rows
    mac('NClaudeGenerations', len(cl))
    mac('NFunctionalClaude', int(cl.functional.sum()))
    q = per_gen[per_gen.model.str.startswith('qwen')]
    mac('NFunctionalQwen', int(q.functional.sum()))
    # residual silent errors of guided Claude workflows, by fault
    epc = ep[ep.model.str.startswith('claude') & (ep.condition == 'C_provenance_guided')]
    resid = epc[epc.outcome == 'silent_error'].fid.value_counts()
    summary['claude_C_silent_by_fault'] = resid.to_dict()
    mac('ClaudeCResidualShareNfourPfour', pct(resid.reindex(['N04', 'P04']).fillna(0).sum() / max(1, resid.sum())))
    mac('ClaudeCSilentTotal', int(resid.sum()))
    # forged manifest (P04): per Claude model x condition silent share (range, not pooled)
    p4 = ep[ep.model.str.startswith('claude') & (ep.fid == 'P04')].groupby(['model', 'condition']).outcome.apply(
        lambda s: (s == 'silent_error').mean())
    mac('PfourSilMin', pct(p4.min(), 0))
    mac('PfourSilMax', pct(p4.max(), 0))
    p4v = ep[ep.model.str.startswith('claude') & (ep.fid == 'P04')]
    mac('PfourValidDetections', int(p4v.valid_abstention.sum()))
    mac('PfourNonSilentClaude', int((p4v.outcome != 'silent_error').sum()))
    heldout_tables(per_gen)

    # condition contrasts within model
    con_rows = []
    for model, g in per_gen.groupby('model'):
        tests = []
        for c1, c2 in (('B_schema_aware', 'A_unassisted'), ('C_provenance_guided', 'A_unassisted'),
                       ('C_provenance_guided', 'B_schema_aware')):
            a = g[g.condition == c1].silent.to_numpy()
            b = g[g.condition == c2].silent.to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            d, lo, hi = bootstrap_diff(a, b)
            p = permutation_test(a, b)
            fa = g[g.condition == c1].false_rejection.to_numpy()
            fb = g[g.condition == c2].false_rejection.to_numpy()
            fd, flo, fhi = bootstrap_diff(fa, fb)
            tests.append({'model': model, 'contrast': f'{ctag[c1]}-{ctag[c2]}', 'silent_diff': d,
                          'silent_diff_lo': lo, 'silent_diff_hi': hi, 'p_perm': p,
                          'fr_diff': fd, 'fr_diff_lo': flo, 'fr_diff_hi': fhi})
        adj = holm([t['p_perm'] for t in tests])
        for t, pa in zip(tests, adj):
            t['p_holm'] = pa
            con_rows.append(t)
    t8 = pd.DataFrame(con_rows)
    t8.to_csv(TAB / 'T8_llm_contrasts.csv', index=False)
    summary['llm_contrasts'] = t8.to_dict('records')
    for t in con_rows:
        if t['model'] in mtag:
            k = mtag[t['model']] + t['contrast'].replace('-', '')
            mac(f'Diff{k}', pct(t['silent_diff']))
            mac(f'DiffLo{k}', pct(t['silent_diff_lo']))
            mac(f'DiffHi{k}', pct(t['silent_diff_hi']))
            mac(f'Red{k}', pct(-t['silent_diff']))         # reduction in points (positive = fewer errors)
            mac(f'RedLo{k}', pct(-t['silent_diff_hi']))
            mac(f'RedHi{k}', pct(-t['silent_diff_lo']))
            mac(f'PHolm{k}', f"{t['p_holm']:.3f}" if t['p_holm'] >= 0.001 else '<0.001')

    # per-fault silent-error share by model x condition
    t9 = ep.groupby(['fid', 'model', 'condition']).outcome.apply(lambda s: (s == 'silent_error').mean()).unstack(
        ['model', 'condition'])
    t9.to_csv(TAB / 'T9_llm_per_fault_silent.csv')
    t9o = ep.groupby(['fid', 'condition']).outcome.value_counts(normalize=True).unstack(fill_value=0)
    t9o.to_csv(TAB / 'T9b_llm_per_fault_outcome_by_condition.csv')
    # faults that are silent in a majority of episodes for every condition (pooled over models)
    maj = ep.groupby(['fid', 'condition']).outcome.apply(lambda s: (s == 'silent_error').mean()).unstack()
    summary['llm_fault_silent_rate_by_condition'] = maj.to_dict('index')

    # consistency: per (model, condition), fraction of (fault) cells where all generations agree on the
    # modal outcome across snapshots; and within-generation snapshot stability.
    cons = []
    for (model, cond), g in ep.groupby(['model', 'condition']):
        modal = g.groupby(['gen_id', 'fid']).outcome.agg(lambda s: s.value_counts().index[0]).unstack()
        agree = (modal.nunique(axis=0) == 1).mean()
        pairs = [((modal.loc[a] == modal.loc[b]).mean()) for a, b in combinations(modal.index, 2)]
        snap_stable = g.groupby(['gen_id', 'fid']).outcome.nunique().eq(1).mean()
        cons.append({'model': model, 'condition': cond, 'faults_all_generations_agree': agree,
                     'mean_pairwise_agreement': float(np.mean(pairs)) if pairs else np.nan,
                     'within_generation_snapshot_stability': snap_stable})
    t10 = pd.DataFrame(cons)
    t10.to_csv(TAB / 'T10_llm_consistency.csv', index=False)
    summary['llm_consistency'] = t10.to_dict('records')
    mac('PairAgreeMin', pct(t10.mean_pairwise_agreement.min()))
    mac('PairAgreeMax', pct(t10.mean_pairwise_agreement.max()))
    mac('SnapStableMin', pct(t10.within_generation_snapshot_stability.min()))

    # compare with deterministic references on the same (replicate-A) episodes
    ba = base_main[(base_main.replicate == 'A') & base_main.workflow.isin(BASE_ORDER)]
    ref = ba.groupby('workflow').outcome.apply(lambda s: (s == 'silent_error').mean())
    summary['baseline_silent_rate_replicateA'] = ref.to_dict()

    rerun = R2 / 'llm' / 'episode_results_test_rerun.csv'
    if rerun.exists():
        rr = pd.read_csv(rerun, keep_default_na=False)
        m = ep.merge(rr, on=['gen_id', 'episode_id'], suffixes=('', '_rerun'))
        same_outcome = (m.outcome == m.outcome_rerun).mean()
        same_metrics = (m.metrics_json == m.metrics_json_rerun).mean()
        diffs = m[m.outcome != m.outcome_rerun]
        summary['llm_reexecution'] = {'pairs': len(m), 'same_outcome': same_outcome, 'same_metrics_json': same_metrics,
                                      'differing': diffs[['gen_id', 'fid', 'outcome', 'outcome_rerun']].head(50)
                                      .to_dict('records')}
        mac('ReexecSameOutcome', pct(same_outcome, 2))
        mac('ReexecPairs', f"{len(m):,}".replace(',', '{,}'))


def heldout_tables(per_gen: pd.DataFrame) -> None:
    """Exploratory held-out scenarios (none described by any prompt); replicate-A snapshots."""
    p = R2 / 'llm' / 'episode_results_heldout.csv'
    if not p.exists():
        return
    ho = pd.read_csv(p, keep_default_na=False)
    assert ho.groupby('gen_id').size().eq(36).all()
    cl = ho[ho.model.str.startswith('claude')]
    pg = cl.groupby(['condition', 'gen_id']).outcome.apply(lambda s: (s == 'silent_error').mean()).reset_index()
    rows = []
    ctag = {'A_unassisted': 'A', 'B_schema_aware': 'B', 'C_provenance_guided': 'C'}
    for cond, g in pg.groupby('condition'):
        m, lo, hi = cluster_bootstrap_mean(g.outcome)
        rows.append({'condition': cond, 'n_generations': len(g), 'silent': m, 'silent_lo': lo, 'silent_hi': hi})
        mac(f'HeldSil{ctag[cond]}', pct(m))
        mac(f'HeldSilLo{ctag[cond]}', pct(lo))
        mac(f'HeldSilHi{ctag[cond]}', pct(hi))
    pd.DataFrame(rows).to_csv(TAB / 'T11_heldout_claude_pooled.csv', index=False)
    byfault = ho.groupby(['model', 'condition', 'fid']).outcome.apply(lambda s: (s == 'silent_error').mean()).unstack()
    byfault.to_csv(TAB / 'T11b_heldout_llm_per_fault.csv')
    b = pd.read_csv(R2 / 'baselines' / 'episode_results.csv', keep_default_na=False)
    hb = b[(b.suite == 'heldout') & b.workflow.str.match(r'B[0-7]_')]
    tb = hb.groupby(['workflow', 'fid']).outcome.apply(lambda s: (s == 'silent_error').mean()).unstack()
    tb.to_csv(TAB / 'T11c_heldout_baselines.csv')
    summary['heldout'] = {'claude_pooled': rows, 'baselines_silent_by_fault': tb.to_dict('index')}
    mac('HeldBfiveSil', pct((hb[hb.workflow == 'B5_contract_manifest'].outcome == 'silent_error').mean()))
    mac('HeldBtwoSwapDet', pct((hb[(hb.workflow == 'B2_statistical') & (hb.fid == 'H02')].outcome
                                 == 'correct_abstention').mean()))
    mac('NHeldExecutions', f"{len(ho):,}".replace(',', '{,}'))
    main = pd.read_csv(R2 / 'llm' / 'episode_results_test.csv', keep_default_na=False)
    f00 = main[main.fid == 'F00'][['gen_id', 'snapshot', 'outcome']].rename(columns={'outcome': 'f00'})
    x = cl.merge(f00, on=['gen_id', 'snapshot'], validate='many_to_one')
    valid = int(((x.outcome == 'correct_abstention') & (x.f00 == 'correct_recovery')).sum())
    summary['heldout']['claude_valid_detections'] = valid
    mac('HeldValidDetections', valid)


def legacy_tables() -> None:
    rows = []
    legacy = (('synthetic', ROOT / 'results' / 'trials.csv'), ('tlc_300_row_excerpt', ROOT / 'results' / 'real' / 'trials.csv'))
    if not all(p.exists() for _, p in legacy):  # the legacy v1 prototype is not in the public release
        return
    for cohort, p in legacy:
        t = pd.read_csv(p)
        for recipe, g in t.groupby('recipe', sort=False):
            per_scn = g.groupby('scenario').status.nunique()
            rows.append({'cohort': cohort, 'recipe': recipe, 'episodes': len(g), 'correct': int(g.correct.sum()),
                         'silent_accept': int(g.silent_accept.sum()),
                         'scenarios_with_seed_varying_outcome': int((per_scn > 1).sum())})
    pd.DataFrame(rows).to_csv(TAB / 'T0_legacy_v1_revalidated.csv', index=False)
    summary['legacy_v1'] = rows


def latex_tables() -> None:
    """Write LaTeX table bodies used verbatim by the manuscript."""
    out = ROOT / 'paper' / 'generated_tables'
    out.mkdir(exist_ok=True)
    esc = lambda t: str(t).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    mode_short = {'consistent': 'cons.', 'original': 'orig.', 'absent': 'none', 'corrupted': 'bad hash',
                  'previous_version': 'v1 for v2', 'wrong_period': 'wrong period', 'forged': 'forged'}
    lines = []
    for f in FAULTS:
        inten = '' if f.default_intensity is None else f' ({100 * f.default_intensity:g}' + r'\%)'
        lines.append(f"{DISPLAY_ID[f.fid]} & {esc(f.name.replace('_', ' '))}{inten} & {'R' if f.desired == 'recover' else 'A'} & "
                     f"{mode_short[f.manifest]} \\\\")
    (out / 'fault_catalog.tex').write_text('\n'.join(lines) + '\n')
    cells = [ln.removesuffix(' \\\\') for ln in lines]
    half = (len(cells) + 1) // 2
    two = [f'{cells[i]} & {cells[i + half]} \\\\' if i + half < len(cells) else f'{cells[i]} & & & & \\\\'
           for i in range(half)]
    (out / 'fault_catalog_2col.tex').write_text('\n'.join(two) + '\n')

    t2 = pd.read_csv(TAB / 'T2_baselines_main.csv')
    label = {'B0_naive': 'B0 naive', 'B1_schema_aware': 'B1 normalizing', 'B2_statistical': 'B2 statistical',
             'B3_contract': 'B3 contract', 'B4_manifest': 'B4 manifest', 'B5_contract_manifest': 'B5 B3+B4',
             'B6_all_checks': 'B6 B5+B2', 'B7_contract_registry': r'B7 B5, indep.\ anchor$^\dagger$'}
    rows = []
    for r in t2.itertuples():
        rows.append(f"{label[r.workflow]} & {100 * r.silent_rate:.1f} [{100 * r.silent_lo:.1f}, {100 * r.silent_hi:.1f}] & "
                    f"{100 * r.false_rejection_rate:.1f} & {100 * r.abstain_detection_rate:.1f} & "
                    f"{r.unverified_accept} & {100 * r.desired_rate:.1f} \\\\")
    (out / 'baselines.tex').write_text('\n'.join(rows) + '\n')

    ab = pd.read_csv(TAB / 'T4_ablation.csv')
    rows = []
    for r in ab.itertuples():
        if r.delta_silent == 0 and r.delta_false_rejection == 0 and r.delta_unverified == 0:
            continue
        fa = esc(str(r.faults_affected).replace('|', ', '))
        sg = lambda v: f'{v:+d}'.replace('-', '$-$')
        rows.append(f"{esc(r.removed_component.replace('_', ' '))} & {sg(r.delta_silent)} & "
                    f"{sg(r.delta_false_rejection)} & {sg(r.delta_unverified)} & {fa} \\\\")
    none = [esc(r.removed_component.replace('_', ' ')) for r in ab.itertuples()
            if r.delta_silent == 0 and r.delta_false_rejection == 0 and r.delta_unverified == 0]
    rows.append(r'\multicolumn{5}{l}{\emph{No change when removed (redundant here):} ' + ', '.join(none)
                + '} ' + '\\\\')
    (out / 'ablation.tex').write_text('\n'.join(rows) + '\n')

    p7 = TAB / 'T7_llm_model_condition.csv'
    if p7.exists():
        t7 = pd.read_csv(p7)
        mname = {'claude-haiku-5-5': 'Haiku 5.5', 'claude-sonnet-5-5': 'Sonnet 5.5', 'claude-opus-5-5': 'Opus 5.5',
                 'qwen2.5-coder-7b-instruct-q4_k_m': 'Qwen2.5-C-7B'}
        rows = []
        order = [m for m in ('claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5') if m in set(t7.model)]
        order += sorted(set(t7.model) - set(order))
        for m in order:
            for c in COND_ORDER:
                r = t7[(t7.model == m) & (t7.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ci = lambda k: f"{100 * r[k]:.1f} [{100 * r[k + '_lo']:.1f}, {100 * r[k + '_hi']:.1f}]"
                rows.append(f"{esc(mname.get(m, m)) if c == COND_ORDER[0] else ''} & {c[0]} & "
                            f"{int(r.functional_generations)}/{int(r.n_generations)} & "
                            f"{ci('silent')} & {ci('false_rejection')} & {100 * r.valid_detection:.1f} & "
                            f"{100 * r.balanced_acc:.1f} & {100 * r.explicit_abstain_share:.0f} \\\\")
            rows.append(r'\midrule')
        (out / 'llm.tex').write_text('\n'.join(rows[:-1]) + '\n')


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    legacy_tables()
    data_tables()
    base_main = baseline_tables()
    llm_tables(base_main)
    timing = R2 / 'timing' / 'full_month_timing.csv'
    if timing.exists():
        tm = pd.read_csv(timing)
        med = tm.groupby('workflow').seconds.median()
        summary['full_month_timing_median_s'] = med.to_dict()
        summary['full_month_rows'] = int(tm.rows.iloc[0])
        mac('FullMonthRows', f"{tm.rows.iloc[0] / 1e6:.2f}")
        mac('TimeNaive', f"{med['B0_naive']:.2f}")
        mac('TimeFull', f"{med['B6_all_checks']:.2f}")
        mac('TimeHashOnly', f"{med['sha256_only']:.2f}")
    clean = R2 / 'verification' / 'clean_reproduction_report.json'
    if clean.exists():
        cr = json.loads(clean.read_text())
        summary['clean_environment_reproduction'] = cr
        mac('CleanSameOutcome', pct(cr['llm_same_outcome_fraction'], 2))
        mac('CleanDiffs', round(cr['llm_pairs_compared'] * (1 - cr['llm_same_outcome_fraction'])))
    latex_tables()
    (R2 / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
    lines = ['% AUTO-GENERATED by pipeshift/analysis.py from episode-level records. Do not edit.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())]
    (ROOT / 'paper' / 'generated_macros.tex').write_text('\n'.join(lines) + '\n')
    print(f'{len(macros)} macros; tables in {TAB}')


if __name__ == '__main__':
    main()
