"""Fetch only the reference images at an immutable source commit."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

COMMIT = 'ecbc2b219606c4153bdec47cd07f6ab00652c6d6'
REPOSITORY = 'nicknochnack/ImageClassification'


def fetch(url):
    for attempt in range(4):
        try:
            request = Request(url,headers={'User-Agent':'cnn-image-classifier'})
            with urlopen(request,timeout=90) as response:
                return response.read()
        except (OSError, HTTPException):
            if attempt == 3:
                raise
            time.sleep(attempt+1)


def blob_hash(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def download(destination):
    destination = Path(destination)
    tree = json.loads(Path(__file__).with_name('reference-tree.json').read_text(encoding='utf-8'))
    if tree['sha'] != COMMIT:
        raise RuntimeError('Manifest does not match the pinned source commit')
    entries = [r for r in tree['tree'] if r['type']=='blob'
               and r['path'].startswith(('data/happy/','data/sad/'))]
    def one(entry):
        relative = Path(entry['path']).relative_to('data')
        target = destination/relative
        if target.exists():
            if blob_hash(target.read_bytes()) != entry['sha']:
                raise ValueError(f'Existing file differs from source: {target}')
            return
        try:
            content = fetch(f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/'+quote(entry['path']))
        except (OSError, HTTPException):
            blob = json.loads(fetch(f'https://api.github.com/repos/{REPOSITORY}/git/blobs/'+entry['sha']))
            content = base64.b64decode(blob['content'])
        if blob_hash(content) != entry['sha']:
            raise ValueError(f'Source checksum mismatch: {relative}')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(one,entry) for entry in entries]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%25 == 0 or i == len(entries):
                print(f'Checked/downloaded {i}/{len(entries)} images',flush=True)
    provenance = {'repository':REPOSITORY,'commit':COMMIT,'files':len(entries),
                  'verification':'Git blob SHA-1 for every downloaded file'}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'source.json').write_text(json.dumps(provenance,indent=2))
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data'))
    download(parser.parse_args().output)
