"""Hash the frozen local weights/configuration once, without importing torch."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
args = parser.parse_args()
base = Path('/data2/wyd/models/Qwen3-VL-8B-Instruct')
checkpoint = Path('/data2/wyd/experiments/cdlsd_vep_v1_formal_50k_s42_20261104-010215/checkpoint-400')
processor = Path('/data1/wyd/models/Qwen3-VL-4B-Instruct')
paths = list(base.glob('model-*.safetensors'))
paths += [base / name for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json', 'chat_template.json', 'vocab.json', 'merges.txt')]
paths += [checkpoint / name for name in ('adapter_model.safetensors', 'adapter_config.json', 'cdlsd_modules.pt', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json')]
paths += [processor / name for name in ('tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json')]
files = {}
for path in paths:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    files[str(path)] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    print(f'[{datetime.now().isoformat(timespec="seconds")}] hashed {path.name}', flush=True)
report = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'manifest_python': platform.python_version(), 'files': files,
          'checkpoint': 'fixed CDLSD checkpoint-400; original training stopped near step 803; latest saved checkpoint-800 retained'}
Path(args.output).write_text(json.dumps(report, indent=2))
