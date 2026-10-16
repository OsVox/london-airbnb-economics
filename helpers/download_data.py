"""Restore exact public snapshots; fail rather than silently accept changed files."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
R=Path(__file__).resolve().parents[1]
EXPECTED={'2026-09-17':'8017870734a9bffa1b424937372e5f02f8c4a114e5060534d7423be5fd718b4c','2026-06-19':'f32fbf1bdd3a4fcdb58ffe901b4ac20d8614f77cd36f1e3bbd18523e6e2acabd'}
for date,digest in EXPECTED.items():
 p=R/f'data/london_{date}.csv.gz';url=f'https://data.insideairbnb.com/united-kingdom/england/london/{date}/data/listings.csv.gz'
 if not p.exists():
  tmp=p.with_suffix('.download');urlretrieve(url,tmp)
  assert hashlib.sha256(tmp.read_bytes()).hexdigest()==digest,f'Changed source file: {url}'
  tmp.rename(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,f'Checksum mismatch: {p}'
 print('Verified',p.name,digest)
