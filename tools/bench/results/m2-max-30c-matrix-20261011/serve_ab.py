"""Paired SSE TTFT with distinct prefixes to exclude cache hits.

Start two normal monolith.serve CLIs, changing only MONOLITH_ACCELERATOR=off/on.
Use --keep-warm 0, the same BF16 pack, capacity 2048, and 128-token chunks.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys

import httpx
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--off-log', type=Path, required=True)
p.add_argument('--on-log', type=Path, required=True)
p.add_argument('--bench-root', type=Path, default=Path(__file__).resolve().parents[2])
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.bench_root))
from serve_stream_latency import measure

key = os.environ.get('LITHOS_METAL_API_KEY') or os.environ.get('LMK_API_KEY') or os.environ.get('MONOLITH_API_KEY')
headers = {'Authorization': 'Bearer ' + key} if key else {}
clients = {label: httpx.Client(timeout=180, headers=headers, base_url=f'http://127.0.0.1:{port}')
           for label, port in [('off', 18091), ('on', 18092)]}
logs = dict(off=a.off_log, on=a.on_log)
tok = Tokenizer.from_file(str(a.model/'tokenizer.json'))
unit = tok.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False).ids
rows = []
for length in [128, 512, 1024]:
    passage = tok.decode((unit * (-(-length // len(unit))))[:length])
    for rep in [-1, 0, 1, 2, 3, 4]:
        content = f'Run {length}:{rep}. Continue this passage with one short sentence.\n' + passage
        payload = dict(model='Qwen3.5-0.8B', messages=[dict(role='user', content=content)],
                       temperature=0, max_completion_tokens=32)
        pair = {}
        for label in (['off','on'] if rep % 2 == 0 else ['on','off']):
            result = measure(clients[label], '/v1/chat/completions', payload)
            metrics = ast.literal_eval([line.split('Generation: ',1)[1] for line in logs[label].read_text().splitlines() if 'Generation: ' in line][-1])
            assert result['ttft_ms'] is not None and result['usage']['completion_tokens'] > 0
            assert metrics['cached_prompt_tokens'] == 0, metrics
            if rep >= 0:
                assert metrics['setup_ms'] == 0, metrics
            pair[label] = dict(**result, server_metrics=metrics)
        assert pair['off']['text'] == pair['on']['text'], (length, rep, pair)
        assert pair['off']['usage'] == pair['on']['usage'], (length, rep, pair)
        row = dict(source_token_slice_length=length, repeat=rep, warmup=rep==-1, samples=pair)
        rows.append(row)
        a.out.write_text(json.dumps(rows, indent=2)+'\n')
        print(length,rep,{label:round(r['ttft_ms'],2) for label,r in pair.items()},flush=True)
for client in clients.values():
    client.close()
