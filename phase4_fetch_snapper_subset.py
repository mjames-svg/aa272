"""Fetch ONLY SnapperGPS JSON fixes and RTK tracks, avoiding 330 MB snapshots.

Uses GitHub's public repository tree, and saves data under consistent filenames.
Run this on your own computer with internet access (the analysis ZIP works offline).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request,urlopen

REPO='JonasBchrt/snapshot-gnss-data-2'
TREE=f'https://api.github.com/repos/{REPO}/git/trees/main?recursive=1'
BASE=f'https://raw.githubusercontent.com/{REPO}/main/'


def read_url(url,max_bytes=25_000_000):
    request=Request(url,headers={'User-Agent':'AA272-GNSS-Research/1.0','Accept':'application/vnd.github+json'})
    with urlopen(request,timeout=60) as response:
        data=response.read(max_bytes+1)
    if len(data)>max_bytes:raise RuntimeError('Data file exceeds safety size limit: '+url)
    return data


def matches_track(filename,track):
    base=filename.lower()
    return bool(re.search(r'(?<![a-z0-9])'+re.escape(track.lower())+r'(?![a-z0-9])',base))


def fetch(out, tracks):
    tree=json.loads(read_url(TREE).decode('utf8'))
    if tree.get('truncated'):raise RuntimeError('GitHub tree truncated; cannot reliably locate files.')
    paths=[x['path'] for x in tree.get('tree',[]) if x.get('type')=='blob' and x['path'].lower().endswith('.json')]
    downloaded=[]
    for track in tracks:
      for folder in ('fixes','ground-truth'):
        options=[p for p in paths if p.startswith(folder+'/') and matches_track(Path(p).name,track)]
        if len(options)!=1:
            raise FileNotFoundError(f'Expected exactly one {folder} JSON for {track}; found {options}. Inspect repository file names.')
        url=BASE+quote(options[0],safe='/')
        body=read_url(url)
        parsed=json.loads(body)
        if not isinstance(parsed,(list,dict)):raise ValueError('Unexpected JSON structure: '+options[0])
        dest=Path(out)/folder/(track+'.json')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(body)
        downloaded.append(dict(track=track,kind=folder,source_path=options[0],bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
        print('Saved',dest,'bytes',len(body))
    (Path(out)/'source_manifest.json').write_text(json.dumps({'repository':f'https://github.com/{REPO}', 'files':downloaded},indent=2))
    return downloaded

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--tracks',nargs='+',default=['w02','w06','c00','c01'],help='Walking/cycling track IDs')
    ap.add_argument('--out',default='external/snapper_reference_subset')
    args=ap.parse_args();fetch(args.out,args.tracks)
