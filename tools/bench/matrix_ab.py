"""Paired accelerator off/on model benchmark, excluding compilation and prefix caching.

Use the same checkpoint, pack, profile and prompt on one device. After warmup,
alternate the two paths and retain every sample; fail on greedy token changes.
Prefill wall time is measured inside Session, independently of HTTP/SSE TTFT.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from monolith.backends.metal import ChipConfig
from monolith.generate import load_session
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[128, 512, 1024])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.repeats < 3 or a.tokens < 2 or not a.lengths or min(a.lengths) < 1:
        p.error('use at least three paired repeats, two generated tokens and positive prompt lengths')
    if os.environ.get('MTL_SHADER_VALIDATION') == '1':
        p.error('run performance measurements without shader validation; validate correctness separately')
    doc = json.loads(a.profile.read_text())
    sessions = {}
    for label in ('off', 'on'):
        cfg = copy.deepcopy(doc)
        cfg['engine']['accelerator'] = label
        profile = ChipConfig.from_dict(cfg['name'], cfg)
        sessions[label] = load_session(str(a.model), str(a.pack), profile=profile,
            max_context=max(a.lengths) + a.tokens + 128, prefill_chunk_size=128,
            eos=-1, autotune=False, prefix_cache=False, accelerator=label)
    unit = Tokenizer.from_file(str(a.model/'tokenizer.json')).encode(
        'The quick brown fox jumps over the lazy dog. ', add_special_tokens=False).ids
    info = sessions['on'].dev.info()
    source_hash = hashlib.sha256()
    for path in sorted([*ROOT.joinpath('monolith').rglob('*.py'), *ROOT.joinpath('kernels').rglob('*.metal')]):
        source_hash.update(path.relative_to(ROOT).as_posix().encode())
        source_hash.update(path.read_bytes())
    metadata = dict(chip=info.name, cores=info.gpu_cores, family=info.apple_family,
        measured=datetime.datetime.now(datetime.timezone.utc).isoformat(), repeats=a.repeats,
        os=platform.mac_ver()[0], python=platform.python_version(), model=a.model.name, pack=a.pack.name,
        profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
        source_sha256=source_hash.hexdigest(),
        pack_manifest_sha256=hashlib.sha256(a.pack.joinpath('manifest.json').read_bytes()).hexdigest(),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        chunk=128, prefix_cache=False, autotune=False, shader_validation=False)
    with a.out.open('w') as out:
        out.write(json.dumps(dict(metadata=metadata))+'\n')
        out.flush()
        for length in a.lengths:
            ids = (unit * (-(-length//len(unit))))[:length]
            expected = None
            for label, session in sessions.items():
                print('warmup', length, label, flush=True)
                r = session.generate(ids, a.tokens)
                if expected is None:
                    expected = r.tokens
                assert r.tokens == expected, (length, label, r.tokens, expected)
            samples = {label: [] for label in sessions}
            for rep in range(a.repeats):
                for label in (list(sessions) if rep%2 == 0 else list(reversed(sessions))):
                    session = sessions[label]
                    t0 = time.perf_counter()
                    r = session.generate(ids, a.tokens)
                    assert r.tokens == expected, (length, label, rep, r.tokens, expected)
                    assert r.setup_ms == 0 and r.cached_prompt_tokens == 0, 'timed run compiled or reused a prefix'
                    samples[label].append(dict(prefill_gpu_ms=r.prefill_ms,
                        prefill_wall_ms=r.prefill_wall_ms, decode_gpu_ms_per_token=r.ms_per_token,
                        decode_wall_ms=r.decode_wall_ms, wall_ms=(time.perf_counter()-t0)*1000,
                        cached_prompt_tokens=r.cached_prompt_tokens, setup_ms=r.setup_ms))
                    print(length, rep, label, samples[label][-1], flush=True)
            row = dict(prompt_tokens=length, generated_tokens=a.tokens, tokens=expected,
                       greedy_equal=True, samples=samples)
            out.write(json.dumps(row)+'\n')
            out.flush()


if __name__ == "__main__":
    main()
