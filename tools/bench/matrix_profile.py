"""Measure matrix crossover at a chip configuration's retained pack layout.

Writes a candidate configuration and raw measurements to --out. Attention,
packing and cross-worker fusion policies remain as supplied by --config.
"""
import argparse
import datetime
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monolith.backends.metal import ChipConfig, get_backend, using_backend
from monolith.backends.metal.calibration import accelerator_plan, cost_key, cost_table
from gemv_bench import Bench
from gemm_bench import GemmBench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shape', default='17408x5120')
    parser.add_argument('--copies', type=int, help='default: approximately 2 GiB streamed per point')
    parser.add_argument('--reps', type=int, default=3)
    args = parser.parse_args()
    if args.reps < 3 or (args.copies is not None and args.copies < 1):
        parser.error('use at least three repetitions and a positive copy count')
    if os.environ.get('MTL_SHADER_VALIDATION') == '1':
        parser.error('measure without shader validation; validate correctness separately')
    doc = json.loads(args.config.read_text())
    config = ChipConfig.from_dict(doc['name'], doc)
    shader, matrix = Bench(), GemmBench()
    get_backend(config.backend).validate_device(config, shader.info)
    n, k = map(int, args.shape.split('x'))
    m = dict(family=config.family, chip=shader.info.name, shape=args.shape, copies=args.copies,
             reps=args.reps, lane_order=config.lane_order, scale_placement=config.scale_placement,
             shader_ms={}, shader_geometry={}, tile_ms={}, raw=[])
    geometry = [('crew', dict(tg_per_core=1)), ('crew2', dict(tg_per_core=2)),
                ('block', dict(one_block_per_sg=True, tg=64))]
    options = dict(rows=16, lane_order=config.lane_order, placement=config.scale_placement,
                   copies=args.copies, reps=args.reps)
    with using_backend(config.backend):
        for fmt in ('nvfp4', 'fp8_e4m3', 'int4_affine', 'bf16', 'int8'):
            m['shader_ms'][fmt], m['shader_geometry'][fmt], m['tile_ms'][fmt] = {}, {}, {}
            for t in (1, 2, 4, 8):
                candidates = []
                for name, geo in geometry:
                    result = shader.run(fmt, n, k, t=t, **options, **geo)
                    if result['ok'] is not True:
                        raise RuntimeError(f'shader oracle failed: {result}')
                    m['raw'].append(result)
                    candidates.append((result['ms'], name))
                best = min(candidates)
                m['shader_ms'][fmt][t], m['shader_geometry'][fmt][t] = best
                print('shader', fmt, t, best, flush=True)
            for tm in (8, 16, 32):
                result = matrix.run(fmt, n, k, tm=tm, **options)
                if result['ok'] is not True:
                    raise RuntimeError(f'matrix oracle failed: {result}')
                m['raw'].append(result)
                m['tile_ms'][fmt][tm] = result['ms']
                print('matrix', fmt, tm, result['tn'], result['tk'], result['ms'], flush=True)
    shader_costs = {cost_key(fmt): cost_table(points) for fmt, points in m['shader_ms'].items()}
    tile_costs = {cost_key(fmt): {tm: round(ms / m['shader_ms'][fmt][1], 3) for tm, ms in points.items()}
                  for fmt, points in m['tile_ms'].items()}
    accelerator, minimum, reason = accelerator_plan(shader_costs, tile_costs)
    doc['engine'].update(accelerator=accelerator, accelerator_min_t=minimum,
                         cost_T={**shader_costs, **{'accelerator_' + fmt: points for fmt, points in tile_costs.items()}})
    doc['validation'] = 'unmeasured'
    doc['note'] = 'Matrix calibration candidate; full-model and serving qualification required.'
    doc['matrix_measurements'] = dict(date=datetime.date.today().isoformat(), os=platform.mac_ver()[0],
        command='python ' + ' '.join(sys.argv), decision=reason, measurements=m)
    args.out.write_text(json.dumps(doc, indent=2) + '\n')
    print('wrote', args.out, reason, flush=True)


if __name__ == '__main__':
    main()
