"""Rebuild the published A100 tables and figures from committed measurements.

This reads existing results only. It never loads models or starts a benchmark.
Install matplotlib separately to generate figures.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path


METHODS = ['none', 'naive', 'dflash', 'dspark', 'jetspec', 'jetspec_graph']
COMPARISON_PROFILE_METHODS = METHODS[:4]
LABELS = dict(none='Baseline', naive='Naive', dflash='DFlash', dspark='DSpark', jetspec='JetSpec eager', jetspec_graph='JetSpec graph')
COLORS = ['#64748b', '#e59b30', '#26856a', '#3975c5', '#9955aa', '#ae4564']
PHASES = ['propose', 'verify', 'ctx_store', 'accept', 'commit']


def phase_value(cell, phase):
    values = cell['metrics']['phase_us_per_committed_token']
    aliases = {'propose': 'propose_tree', 'verify': 'verify_tree', 'accept': 'accept_tree', 'commit': 'commit_tree'}
    key = aliases.get(phase, phase) if cell['job']['method'] == 'jetspec' else ('commit_copy' if phase == 'commit' else phase)
    return values.get(key)


def load(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '|' + '|'.join(['---'] * len(headers)) + '|'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def profile_summary(trace):
    events = trace['traceEvents']
    steps = [e for e in events if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('ProfilerStep#')]
    start = min(e['ts'] for e in steps)
    end = max(e['ts'] + e['dur'] for e in steps)
    kernels = [e for e in events if e.get('cat') == 'kernel']
    device = [e for e in events if e.get('cat') in ['kernel', 'gpu_memcpy', 'gpu_memset']]
    intervals = merged([(max(start, e['ts']), min(end, e['ts'] + e['dur']))
        for e in device if e['ts'] < end and e['ts'] + e['dur'] > start])
    groups = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        groups[e['name']][0] += 1
        groups[e['name']][1] += e['dur']
    ranked = sorted(groups.items(), key=lambda pair: -pair[1][1])
    sync = [e for e in events if e.get('cat') == 'cuda_runtime' and
        e.get('name') in ['cudaDeviceSynchronize', 'cudaStreamSynchronize', 'cudaEventSynchronize']]
    stats = dict(captured_steps=len(steps), window_ms=(end-start)/1000,
        kernel_launches=len(kernels), kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
        device_active_union_ms=sum(b-a for a, b in intervals)/1000,
        device_activity_coverage_pct=100*sum(b-a for a, b in intervals)/(end-start),
        memcpy_events=sum(e.get('cat') == 'gpu_memcpy' for e in events),
        host_sync_calls=len(sync), host_sync_ms=sum(e['dur'] for e in sync)/1000,
        top_kernels=[dict(name=name, calls=v[0], kernel_ms=v[1]/1000) for name, v in ranked[:10]])
    return stats, (start, end, intervals)


def figures(run, cells, profiles, timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'savefig.bbox': 'tight'})
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, concurrency in zip(axes, [1, 8, 32]):
        base = cells[(concurrency, 'none')]['wall_tok_s']
        values = [cells[(concurrency, m)]['wall_tok_s'] for m in METHODS]
        ax.bar(range(len(METHODS)), values, color=COLORS)
        ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=28, ha='right')
        ax.set_title(f'Concurrency {concurrency}')
        ax.set_ylabel('Delivered output tokens / s')
        ax.set_ylim(0, max(values) * 1.25)
        for i, value in enumerate(values):
            ax.text(i, value+max(values)*.025, f'{value:.1f}\n{value/base:.2f}x', ha='center', fontsize=9)
    fig.suptitle('A100 80GB | Qwen3-8B | one seed, one pass')
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(out / f'throughput.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, concurrency in zip(axes, [1, 32]):
        bottom = np.zeros(len(METHODS) - 1)
        for phase in PHASES:
            values = [phase_value(cells[(concurrency, m)], phase) or 0 for m in METHODS[1:]]
            ax.bar(range(len(METHODS) - 1), values, bottom=bottom, label=phase)
            bottom += np.array(values)
        ax.set_xticks(range(len(METHODS) - 1), [LABELS[m] for m in METHODS[1:]])
        ax.set_title(f'Concurrency {concurrency}')
        ax.set_ylabel('CUDA event elapsed us / committed decode token')
    axes[1].legend(fontsize=8)
    fig.suptitle('Unprofiled performance cells: phase event intervals include launch gaps')
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(out / f'phases.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(COMPARISON_PROFILE_METHODS), 1, figsize=(11, 6.5), sharex=True)
    for ax, method, color in zip(axes, COMPARISON_PROFILE_METHODS, COLORS):
        start, end, intervals = timeline[method]
        ax.broken_barh([(0, (end-start)/1000)], (0, 1), facecolors='#e5e7eb')
        ax.broken_barh([((a-start)/1000, (b-a)/1000) for a, b in intervals], (.1, .8), facecolors=color)
        ax.set_yticks([.5], [LABELS[method]])
        ax.set_ylim(-.2, 1.2)
        ax.spines['left'].set_visible(False)
        ax.text(1.005, .5, f"{profiles[method]['device_activity_coverage_pct']:.1f}%", transform=ax.transAxes)
    axes[-1].set_xlabel('Elapsed ms from first captured CPU step (each method starts at zero)')
    fig.suptitle('C=1 diagnostic profiles: CUDA kernel/copy/memset interval union\n8 captured CPU steps per method; coverage is not hardware utilization')
    fig.tight_layout()
    fig.savefig(out / 'timeline.png', dpi=180)
    plt.close(fig)

    for name in ('throughput', 'phases'):
        path = out / f'{name}.svg'
        lines = path.read_text(encoding='utf-8').splitlines()
        path.write_text('\n'.join(line.rstrip() for line in lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('reports/a100-20261118'))
    args = parser.parse_args()
    run = args.run
    cells = {}
    rows = []
    for path in sorted((run / 'data/cells').glob('*.json')):
        c = load(path)
        j, m = c['job'], c['metrics']
        cells[(j['concurrency'], j['method'])] = c
    jetspec = load(run / 'data/jetspec.json')
    for cell in jetspec['cells']:
        job = cell['job']
        method = 'jetspec_graph' if job['jetspec_cuda_graph'] else 'jetspec'
        cells[(job['concurrency'], method)] = cell
    for concurrency in [1, 8, 32]:
        baseline = cells[(concurrency, 'none')]['wall_tok_s']
        for method in METHODS:
            c = cells[(concurrency, method)]
            m = c['metrics']
            rows.append(dict(method=method, concurrency=concurrency, seed=29, passes=1,
                wall_tok_s=c['wall_tok_s'], speedup=c['wall_tok_s']/baseline,
                steady_tok_s=m['steady_tok_s'], ttft_p50_ms=m['ttft_ms']['p50'],
                ttft_p99_ms=m['ttft_ms']['p99'], latency_p50_ms=m['latency_ms']['p50'],
                latency_p99_ms=m['latency_ms']['p99'], tpot_p50_ms=m['tpot_ms']['p50'],
                gap_p99_ms=m['delivery_gap_ms']['p99'], goodput_requests_s=m['goodput_requests_s'],
                peak_allocated_gib=c['peak_allocated_bytes']/2**30,
                peak_reserved_gib=c['peak_reserved_bytes']/2**30,
                preemptions=m['preemptions'], recomputed_tokens=m['recomputed_tokens'],
                acceptance_rate=c['acceptance_rate'],
                committed_tokens_per_sequence_round=m['committed_tokens_per_sequence_round'],
                verified_positions_per_committed_token=m['verified_positions_per_committed_token'],
                target_kv_peak_pct=100*m['peak_target_blocks']/m['target_block_capacity']))
    write_csv(run / 'metrics.csv', rows)
    profiles, timelines = {}, {}
    for method in COMPARISON_PROFILE_METHODS:
        profiles[method], timelines[method] = profile_summary(load(run / 'profiling' / method / 'trace.json.gz'))
        profiles[method]['method'] = method
        dump(run / 'profiling' / method / 'summary.json', profiles[method])
    for name, summary in jetspec['profiles'].items():
        _, mode, concurrency = name.split('-')
        label = f'JetSpec {mode} C={concurrency[1:]}'
        profiles[label] = dict(summary, method=label)
    write_csv(run / 'profiling.csv', [
        {k: v for k, v in summary.items() if k not in ('top_kernels', 'status', 'cudaLaunchKernel', 'cudaGraphLaunch')}
        for summary in profiles.values()
    ])
    figures(run, cells, profiles, timelines)
    speed_table = table(['Method', 'C=1 tok/s (speedup)', 'C=8 tok/s (speedup)', 'C=32 tok/s (speedup)'],
        [[LABELS[method]] + [f"{cells[(c,method)]['wall_tok_s']:.2f} ({cells[(c,method)]['wall_tok_s']/cells[(c,'none')]['wall_tok_s']:.2f}x)" for c in [1,8,32]] for method in METHODS])
    infra_table = table(['C', 'Method', 'Steady tok/s', 'TTFT p50/p99 ms', 'Latency p50/p99 ms', 'TPOT p50 ms', 'Gap p99 ms', 'Goodput req/s', 'Peak alloc GiB'],
        [[r['concurrency'], LABELS[r['method']], f"{r['steady_tok_s']:.2f}", f"{r['ttft_p50_ms']:.2f}/{r['ttft_p99_ms']:.2f}",
          f"{r['latency_p50_ms']:.2f}/{r['latency_p99_ms']:.2f}", f"{r['tpot_p50_ms']:.2f}", f"{r['gap_p99_ms']:.2f}",
          f"{r['goodput_requests_s']:.3f}", f"{r['peak_allocated_gib']:.2f}"] for r in rows])
    algorithm_table = table(['C', 'Method', 'Accepted/proposed', 'Committed tokens/sequence round', 'Verified positions/committed token', 'Target KV peak %', 'Preemptions'],
        [[r['concurrency'], LABELS[r['method']], 'N/A' if r['method']=='none' else f"{r['acceptance_rate']:.3f}",
          f"{r['committed_tokens_per_sequence_round']:.3f}", 'N/A' if r['method']=='none' else f"{r['verified_positions_per_committed_token']:.3f}",
          f"{r['target_kv_peak_pct']:.2f}", r['preemptions']] for r in rows])
    phase_table = table(['C', 'Method'] + [f'{p} us/token' for p in PHASES],
        [[c, LABELS[m]] + ['N/A' if phase_value(cells[(c,m)], p) is None else f"{phase_value(cells[(c,m)], p):.2f}" for p in PHASES] for c in [1,32] for m in METHODS[1:]])
    profile_table = table(['Method', 'CPU steps', 'Window ms', 'Kernel launches', 'Kernel sum ms', 'Device interval union ms', 'Coverage %', 'Memcpy events', 'Host sync ms'],
        [[LABELS.get(m, m), profiles[m]['captured_steps'], f"{profiles[m]['window_ms']:.2f}", profiles[m]['kernel_launches'],
          f"{profiles[m]['kernel_sum_ms']:.2f}", f"{profiles[m]['device_active_union_ms']:.2f}", f"{profiles[m]['device_activity_coverage_pct']:.2f}",
          profiles[m]['memcpy_events'], f"{profiles[m]['host_sync_ms']:.2f}"] for m in profiles])
    kernel_table = table(['Method', 'Top kernel (name shortened)', 'Calls', 'Kernel ms'],
        [[LABELS.get(m, m), '`'+k['name'][:90].replace('|','/')+'`', k['calls'], f"{k['kernel_ms']:.2f}"] for m in profiles for k in profiles[m]['top_kernels'][:3]])
    models = load(Path('benchmarks/models.json'))
    model_table = table(['Role', 'Checkpoint', 'Revision'], [[role, v['repo'], '`'+v['revision']+'`'] for role,v in models.items()])
    tables = dict(speed=speed_table, infra=infra_table, algorithm=algorithm_table, phases=phase_table, profiles=profile_table, kernels=kernel_table, models=model_table)
    dump(run / 'tables.json', tables)
    (run / 'TABLES.md').write_text('\n\n'.join(['# Measured tables'] + [f'## {k}\n\n{v}' for k,v in tables.items()])+'\n', encoding='utf-8')
    print('Built tables, CSV summaries and figures from 18 cells and 8 profiles.')


if __name__ == '__main__':
    main()
