"""Deterministic acquisition of official NYC TLC Yellow Taxi monthly Parquet files.

Downloads each month from the TLC's published CloudFront distribution (linked from
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), records HTTP metadata,
size, SHA-256, record count and schema, and writes data/source_manifest.json.

Files are stored under data/raw/ and are NOT redistributed in the reproducibility
package. Re-running the script verifies existing files against the recorded hashes;
an upstream revision (hash change) fails closed unless --accept-revision is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests

BASE_URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet'
LANDING_PAGE = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
REFERENCE_MONTHS = ('2024-12',)  # prior period used only as stale/reference input
STUDY_MONTHS = tuple(f'2025-{m:02d}' for m in range(1, 13))
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(month: str, dest: Path) -> dict:
    url = BASE_URL.format(month=month)
    head = requests.head(url, timeout=60)
    head.raise_for_status()
    meta = {'month': month, 'url': url,
            'http_last_modified': head.headers.get('Last-Modified'),
            'http_etag': head.headers.get('ETag', '').strip('"'),
            'http_content_length': int(head.headers.get('Content-Length', -1))}
    if not dest.exists() or dest.stat().st_size != meta['http_content_length']:
        tmp = dest.with_suffix('.part')
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with tmp.open('wb') as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        meta['retrieved_utc'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    return meta


def describe(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {'num_rows': pf.metadata.num_rows,
            'columns': [{'name': f.name, 'type': str(f.type)} for f in schema]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', type=Path, default=ROOT / 'data' / 'raw')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'data' / 'source_manifest.json')
    ap.add_argument('--accept-revision', action='store_true')
    args = ap.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.manifest.exists():
        previous = {f['month']: f for f in json.loads(args.manifest.read_text())['files']}
    files = []
    for month in REFERENCE_MONTHS + STUDY_MONTHS:
        dest = args.raw_dir / f'yellow_tripdata_{month}.parquet'
        meta = download(month, dest)
        meta['role'] = 'reference_prior_period' if month in REFERENCE_MONTHS else 'study_period'
        meta['local_file'] = f'data/raw/{dest.name}'
        meta['sha256'] = sha256_file(dest)
        meta['size_bytes'] = dest.stat().st_size
        meta.update(describe(dest))
        old = previous.get(month)
        if old:
            meta.setdefault('retrieved_utc', old.get('retrieved_utc'))
            if old['sha256'] != meta['sha256'] and not args.accept_revision:
                print(f'UPSTREAM REVISION for {month}: {old["sha256"]} -> {meta["sha256"]}', file=sys.stderr)
                return 2
        print(f"{month}: {meta['num_rows']:>9,} rows  sha256={meta['sha256'][:16]}...")
        files.append(meta)
    manifest = {
        'dataset': 'NYC TLC Yellow Taxi Trip Records (monthly Parquet)',
        'publisher': 'New York City Taxi and Limousine Commission (TLC)',
        'landing_page': LANDING_PAGE,
        'data_dictionary': 'https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf',
        'terms_note': ('TLC states trip data are submitted by technology providers and accuracy/completeness '
                       'is not guaranteed. Raw files are not redistributed by this project; use this script.'),
        'manifest_written_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'files': files,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
