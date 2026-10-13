"""Run on the server to prepare (not launch) the missing combination runs.

Uses the original experiment snapshot, splits, seeds and training settings.
Completed runs are reused by symlink; failed runs get fresh output directories.
"""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def patch_model(source):
    start = source.index('    local = Conv1D', source.index('def _sequence_state_tcn_decoder('))
    end = source.index('    if use_transformer:', start)
    tcn = source[start:end]
    replacement = '''    if os.environ.get("LFTS_PAPER_NO_TCN", "0") == "1":
        if not use_transformer:
            raise ValueError("no_tcn requires Transformer; use epoch_only for neither")
        x = Dense(filters, activation="relu", name="v3_state_input_projection")(features)
        x = LayerNormalization(name="v3_state_input_norm")(x)
    else:
'''
    replacement += ''.join('    ' + line if line.strip() else line for line in tcn.splitlines(True))
    result = source[:start] + replacement + source[end:]
    ast.parse(result)
    return result


def prepare(root):
    old = root / 'outputs/paper_ablation_20260918'
    new = root / 'outputs/paper_combinations_20260924'
    if new.exists():
        raise FileExistsError(f'Refusing to overwrite {new}')
    protocols = json.loads((old / 'protocols.json').read_text())
    for protocol in protocols.values():
        actual = hashlib.sha256(Path(protocol['path']).read_bytes()).hexdigest()
        if actual != protocol['sha256']:
            raise ValueError('Original split has changed')
    model = patch_model((old / 'code/src/model_training.py').read_text())
    supervisor = (old / 'supervisor.py').read_text()
    a = supervisor.index('variants=')
    b = supervisor.index('common=', a)
    supervisor = supervisor[:a] + '''variants={'transformer_only':{'LFTS_PAPER_NO_TCN':'1'},
          'full':{}, 'epoch_only':{'LFTS_PAPER_EPOCH_ONLY':'1'},
          'no_transformer':{'LFTS_SEQUENCE_USE_TRANSFORMER':'0'},
          'raw_only':{'LFTS_PAPER_VIEW':'raw'},
          'segment_only':{'LFTS_PAPER_VIEW':'segment'}}
''' + supervisor[b:]
    # Explicit reset avoids inheriting a no-TCN toggle into other runs.
    supervisor = supervisor.replace("'LFTS_PAPER_VIEW':'both','LFTS_PAPER_EPOCH_ONLY':'0'",
                                    "'LFTS_PAPER_VIEW':'both','LFTS_PAPER_EPOCH_ONLY':'0','LFTS_PAPER_NO_TCN':'0'")
    ast.parse(supervisor)
    supervisor = supervisor.replace("lane,'sleepedf','0,1'", "lane,'sleepedf','4,5'")
    supervisor = supervisor.replace("lane,'isruc','2,3'", "lane,'isruc','6,7'")
    preflight = (old / 'preflight.py').read_text()
    preflight = preflight.replace("os.environ['CUDA_VISIBLE_DEVICES']='6'",
                                  "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')")
    preflight = preflight.replace("suite=root/'outputs/paper_ablation_20260918'",
                                  "suite=Path(__file__).resolve().parent")
    preflight = preflight.replace("'epoch_only','no_film'):", "'epoch_only','no_film','transformer_only'):")
    preflight = preflight.replace("    kw={**kwargs", "    os.environ['LFTS_PAPER_NO_TCN']=str(int(variant=='transformer_only'))\n    kw={**kwargs")
    preflight = preflight.replace("    check=None", '''    if variant == 'transformer_only':
        assert not any(n.startswith('v3_state_tcn_') for n in names)
        assert 'v3_state_local_conv' not in names and 'v3_state_wide_conv' not in names
        assert any(isinstance(layer, tf.keras.layers.MultiHeadAttention) for layer in m.layers)
    check=None''')
    ast.parse(preflight)
    new.mkdir()
    shutil.copytree(old / 'code', new / 'code', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', 'outputs'))
    (new / 'code/src/model_training.py').write_text(model)
    (new / 'supervisor.py').write_text(supervisor)
    (new / 'preflight.py').write_text(preflight)
    shutil.copy2(old / 'protocols.json', new / 'protocols.json')
    (new / 'runs').mkdir()
    reused = []
    for run in sorted((old / 'runs').iterdir()):
        status = run / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('state') != 'completed':
            continue
        if '_no_film_' in run.name:
            continue
        if not (run / 'metrics.json').exists():
            raise ValueError(f'Missing metrics for completed run: {run}')
        (new / 'runs' / run.name).symlink_to(run, target_is_directory=True)
        reused.append(run.name)
    (new / 'preparation.json').write_text(json.dumps({
        'reused': reused, 'seeds': [42, 2024, 2025],
        'note': 'Not launched. Inspect GPUs and old supervisor, run preflight, then launch.',
        'new_variant': 'Dual view, per-epoch projection, Transformer; no temporal convolutions',
    }, indent=2))
    print(json.dumps({'prepared': str(new), 'reused': reused}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/b502/wssyi/LIME-for-Time-Series'))
    args = parser.parse_args()
    prepare(args.root)
