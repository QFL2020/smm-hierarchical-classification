"""Portable locations: data and generated outputs stay outside the upload tree."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def data_path():
    value = os.environ.get('SMM_DATA_CSV')
    if not value: raise RuntimeError('Set SMM_DATA_CSV to your authorized prepared CSV.')
    p = Path(value).expanduser().resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    return p
def output_root():
    value = os.environ.get('SMM_OUTPUT_ROOT')
    if not value: raise RuntimeError('Set SMM_OUTPUT_ROOT to a private output directory outside the repository.')
    p = Path(value).expanduser().resolve()
    if p == ROOT or ROOT in p.parents: raise ValueError('Keep study outputs outside the public source repository.')
    return p
def new_output(name):
    p = output_root()/name
    if p.exists() and any(p.iterdir()): raise FileExistsError(f'Refusing to overwrite {p}')
    p.mkdir(parents=True,exist_ok=True)
    return p
def feature_names():
    import json
    f=json.loads(Path(__file__).with_name('features.json').read_text())
    return f['categorical_vars'], f['continuous_vars']
