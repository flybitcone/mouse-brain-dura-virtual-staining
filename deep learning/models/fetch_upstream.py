"""Vendor pinned official model definitions and their licenses (no weights)."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent


def fetch(url):
    return urlopen(Request(url, headers={'User-Agent':'virtual-staining-benchmark'}), timeout=60).read()


if __name__ == '__main__':
    records = []
    for repo, source, license_name, local in [
        ('swz30/Restormer', 'basicsr/models/archs/restormer_arch.py', 'LICENSE.md', 'restormer'),
        ('JingyunLiang/SwinIR', 'models/network_swinir.py', 'LICENSE', 'swinir')]:
        sha = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
        for remote, output in [(source, local+'_upstream.py'), (license_name, local+'_LICENSE.txt')]:
            url = f'https://raw.githubusercontent.com/{repo}/{sha}/{remote}'
            data = fetch(url)
            (ROOT / output).write_bytes(data)
            records.append(dict(repository=repo, commit=sha, source=url, local=output, sha256=hashlib.sha256(data).hexdigest()))
    (ROOT / 'upstream_sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Saved two official architecture definitions, licenses and pinned source hashes.')
