"""Run six JetSpec performance cells, one seed/pass, then separate profiles.

Original serial JetSpec and baseline numbers are historical references from the
completed five-method run. No extra seeds, repeated cells or correctness gates.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prior = {}
    for file in args.source.glob('cell-*.json'):
        cell = json.loads(file.read_text())
        if cell['job']['method'] in ('none', 'jetspec'):
            prior[(cell['job']['concurrency'], cell['job']['method'])] = cell
    jobs = []
    # C=8/32 expose batching before the longer latency-oriented C=1 cells.
    for concurrency in [8, 32, 1]:
        for graphs in [False, True]:
            job = dict(prior[(concurrency, 'jetspec')]['job'])
            job.update(jetspec_backend='triton', jetspec_cuda_graph=graphs)
            jobs.append(job)
    manifest = dict(seed=29, passes=1, historical_source=str(args.source), jobs=jobs)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for index, job in enumerate(jobs):
        name = f"triton-{'graph' if job['jetspec_cuda_graph'] else 'eager'}-c{job['concurrency']}"
        job_file = args.out/f'{name}.job.json'
        result_file = args.out/f'{name}.json'
        job_file.write_text(json.dumps(job, indent=2))
        print(f'[{index+1}/{len(jobs)}] {name}', flush=True)
        if not result_file.exists():
            with (args.out/f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'bench_spec.py'), 'worker',
                    '--job', str(job_file), '--out', str(result_file)], stdout=log, stderr=subprocess.STDOUT, check=True)
        cell = json.loads(result_file.read_text())
        print(f"  {cell['wall_tok_s']:.2f} tok/s; stats={cell['stats']}", flush=True)
    rows = []
    lines = ['# JetSpec execution optimization', '',
        'One seed (29), one pass, same frozen inputs and sampling settings. Historical baseline/reference cells',
        'come from job 24846937; optimized cells are newly measured. GPU UUID/driver are recorded for comparison.', '',
        '| C | backend | graphs | tok/s | vs historical AR | vs serial tree | TTFT p99 ms | TPOT p50 ms | peak GiB |',
        '|---|---|---|---|---|---|---|---|---|']
    for c in [1,8,32]:
        reference, baseline = prior[(c,'jetspec')], prior[(c,'none')]
        results = [('reference',False,reference)]
        for graph in [False,True]:
            name = f"triton-{'graph' if graph else 'eager'}-c{c}"
            results.append(('triton',graph,json.loads((args.out/f'{name}.json').read_text())))
        for backend, graph, cell in results:
            m = cell['metrics']
            row = dict(concurrency=c, backend=backend, graphs=graph, wall_tok_s=cell['wall_tok_s'],
                vs_historical_ar=cell['wall_tok_s']/baseline['wall_tok_s'],
                vs_serial_tree=cell['wall_tok_s']/reference['wall_tok_s'],
                ttft_p99_ms=m['ttft_ms']['p99'], tpot_p50_ms=m['tpot_ms']['p50'],
                peak_gib=cell['peak_allocated_bytes']/2**30)
            rows.append(row)
            lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row.values())+' |')
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (args.out/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Profile each optimized execution mode at C=1 and C=32, separately from timing.
    for c in [1,32]:
        for graph in [False,True]:
            name=f"triton-{'graph' if graph else 'eager'}-c{c}"
            directory=args.out/'profiles'/name
            if (directory/'result.json').exists():
                continue
            job=json.loads((args.out/f'{name}.job.json').read_text())
            job.update(num_requests=max(4,c),max_tokens=64)
            path=args.out/f'{name}.profile-job.json';path.write_text(json.dumps(job,indent=2))
            print(f'PROFILE {name}',flush=True)
            subprocess.run([sys.executable,str(ROOT/'profile_spec.py'),'--job',str(path),'--out',str(directory)],check=True)
    inputs=args.out/'inputs';inputs.mkdir(exist_ok=True)
    for suffix in ['.json','.metadata.json']:
        shutil.copyfile(ROOT/f'benchmarks/inputs/speed-128{suffix}', inputs/f'speed-128{suffix}')
    print(f'COMPLETE {args.out}',flush=True)


if __name__=='__main__':
    main()
