#!/usr/bin/env bash
# Fetch ONLY the FutureBench videos the eval reads out of haonan3/V1-33K.
#
# Facts this script is built on (verified against the HF HTTP API at the pinned
# revision, see the run's research.md):
#   * futurebench.json has 1056 rows, each with
#     "video_path": "first_part_video/<source>/<clip>.(mp4|mkv)" (both .mp4 and
#     .mkv occur, so nothing here may assume .mp4).
#   * The repo ships clips ONLY inside 70 gzip archives
#     first_segment_video/first_segement_video_data_part0..69.tar.gz (~155.9 GB);
#     there is no per-clip file, no plain .tar, and no published member manifest,
#     so the archive -> members mapping can only be learned by streaming.
#   * The archives are gzip streams without random access (mode='r|gz'): reaching
#     a wanted member costs every byte that precedes it inside that archive.
#     Therefore network != disk by construction -- a first pass pulls whole
#     archives (~all of the first-segment corpus, every archive holds some
#     referenced clip) while DISK only grows by the ~1e3 referenced clips. The
#     per-archive index written under <dest>/.fetch_state/ makes later runs
#     genuinely skip archives, including the ones that hold nothing referenced.
#   * Archive dir is first_segment_video/ but the JSON paths (and the tar member
#     names) are first_part_video/<source>/... , hence the basename fallback.
#
# Streaming, atomic-write and resume logic mirrors the proven
# scripts/prepare_dual_benchmarks.py (tarfile 'r|gz', '.partial' + exact size
# check, per-archive resume marker, 2 GiB disk floor). That file is not modified;
# the additions here are a stdlib-only HTTP inventory (no huggingface_hub) and
# reporting of retained clips / bytes actually pulled.
#
# Usage: bash scripts/fetch_futurebench_data.sh --dest DIR [--dry-run]
#   --dest DIR   output dir; gets futurebench.json, first_part_video/**, and the
#                eval-ready futurebench_local.json (absolute video_path entries)
#   --dry-run    print the plan and the byte budget, download no archive
# Env: HF_ENDPOINT (default https://huggingface.co), HF_TOKEN (sent only to the
#      endpoint host), PYTHON (interpreter, default python3)
set -euo pipefail

DEST=""
DRY_RUN=0
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -32; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="${2:?--dest needs a directory}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (only --dest DIR, --dry-run, -h)" >&2; exit 2 ;;
  esac
done
[[ -n "${DEST}" ]] || { echo "--dest DIR is required (json + clips are written there)" >&2; exit 2; }
PY="${PYTHON:-python3}"
command -v "${PY}" >/dev/null 2>&1 || { echo "no ${PY} on PATH; set PYTHON=/path/to/python3" >&2; exit 1; }

exec "${PY}" - "${DEST}" "${DRY_RUN}" <<'PYEOF'
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'haonan3/V1-33K'
REV = 'd8c91f2df9aeec5ee514aa1f97efd3707bb88db8'
API = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
ARCHIVE_DIR = 'first_segment_video'
VIDEO_DIR = 'first_part_video'
JSON_NAME = 'futurebench.json'
N_ROWS = 1056
N_ARCHIVES = 70
MIN_FREE = 2 * 2 ** 30          # never fill the shared disk while streaming
UA = 'future-l1-fetch-futurebench/1.0'

DEST = Path(sys.argv[1]).resolve()
DRY_RUN = sys.argv[2] == '1'
STATE = DEST / '.fetch_state'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # resolve/ redirects to a CDN host; following by hand keeps HF_TOKEN from
    # being replayed to that other host.
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(_NoRedirect)
HOST = urllib.parse.urlparse(API).netloc


def headers(url):
    head = {'User-Agent': UA}
    token = os.environ.get('HF_TOKEN')
    if token and urllib.parse.urlparse(url).netloc == HOST:
        head['Authorization'] = 'Bearer ' + token
    return head


def open_stream(url, timeout=120):
    for _ in range(6):
        request = urllib.request.Request(url, headers=headers(url))
        try:
            return OPENER.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get('Location') if exc.headers else None
            if exc.code in (301, 302, 303, 307, 308) and location:
                url = urllib.parse.urljoin(url, location)
                continue
            raise
    raise RuntimeError('too many redirects')


def resolve_url(path):
    return '{}/datasets/{}/resolve/{}/{}'.format(API, REPO, REV, urllib.parse.quote(path))


def save(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def download(url, target, size):
    partial = Path(str(target) + '.partial')
    with open_stream(url) as stream, partial.open('wb') as out:
        shutil.copyfileobj(stream, out, length=1024 * 1024)
    if partial.stat().st_size != size:
        raise RuntimeError('{}: got {} bytes, expected {}'.format(target.name, partial.stat().st_size, size))
    partial.replace(target)


class CountingReader:
    """Counts bytes actually pulled off the network (an early exit skips the tail)."""

    def __init__(self, stream):
        self.stream = stream
        self.n = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.n += len(data)
        return data

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        buffer[:len(data)] = data
        self.n += len(data)
        return len(data)

    def seekable(self):
        return False

    def readable(self):
        return True

    def close(self):
        self.stream.close()

    @property
    def closed(self):
        return self.stream.closed


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)

    tree = json.load(open_stream('{}/api/datasets/{}/tree/{}?recursive=true'.format(API, REPO, REV)))
    files = {e['path']: e for e in tree if e.get('type') == 'file'}
    archives = sorted((e for p, e in files.items() if p.startswith(ARCHIVE_DIR + '/')
                       and p.endswith('.tar.gz') and e.get('size')), key=lambda e: e['path'])
    if len(archives) != N_ARCHIVES:
        raise SystemExit('expected {} archives under {}/, found {}; refusing to guess'
                         .format(N_ARCHIVES, ARCHIVE_DIR, len(archives)))
    archive_bytes = sum(e['size'] for e in archives)

    # futurebench.json (2.7 MB) is the only per-row source of truth.
    meta = files[JSON_NAME]
    local_json = DEST / JSON_NAME
    if not (local_json.is_file() and local_json.stat().st_size == meta['size']):
        print('downloading {} ({} bytes)'.format(JSON_NAME, meta['size']), flush=True)
        download(resolve_url(JSON_NAME), local_json, meta['size'])
    rows = json.loads(local_json.read_text())
    if len(rows) != N_ROWS:
        raise SystemExit('{}: {} rows, expected {}'.format(JSON_NAME, len(rows), N_ROWS))

    # The eval resolves absolute video_path entries verbatim
    # (lmms_eval/tasks/futurebench/utils.py), so keep the JSON-relative layout.
    wanted = []
    for row in rows:
        rel = row['video_path'].removeprefix('./')
        if not rel.startswith(VIDEO_DIR + '/') or '..' in Path(rel).parts:
            raise SystemExit('unexpected video_path: {!r}'.format(rel))
        wanted.append(rel)
    distinct = sorted(set(wanted))
    wanted_set = set(distinct)
    by_name = {}
    for rel in distinct:
        by_name.setdefault(Path(rel).name, []).append(rel)
    pending = {rel for rel in distinct if not (DEST / rel).is_file()}

    save(DEST / 'futurebench_download_plan.json', {
        'repo': REPO, 'revision': REV, 'n_questions': len(rows), 'n_video_refs': len(wanted),
        'n_distinct_videos': len(distinct), 'n_archives': len(archives),
        'network_bytes_if_all_archives_scanned': archive_bytes, 'archive_dir': ARCHIVE_DIR,
        'video_prefix': VIDEO_DIR + '/',
        'note': 'gzip archives have no member index: a first pass pulls whole archives to reach '
                'the referenced members, so network bytes >> retained bytes.'})
    print('rows={} video refs={} distinct videos={} already local={} missing={}'
          .format(len(rows), len(wanted), len(distinct), len(distinct) - len(pending), len(pending)))
    print('archives={} total={} bytes ({:.1f} GiB) -> upper bound on first-pass network'
          .format(len(archives), archive_bytes, archive_bytes / 2 ** 30))

    if DRY_RUN:
        if not pending:
            print('DRY_RUN: every referenced clip is already present; nothing to download')
        else:
            print('DRY_RUN: nothing downloaded; {} clips missing, network budget <= {:.1f} GiB '
                  '(the archive->member map is only knowable by streaming)'
                  .format(len(pending), archive_bytes / 2 ** 30))
        return 0

    streamed = scanned = skipped_resume = skipped_empty = with_needs = bytes_read = 0
    for entry in archives:
        if not pending:
            print('all referenced clips are on disk; stopping before {}'.format(entry['path']), flush=True)
            break
        name = entry['path']
        marker = STATE / (Path(name).name + '.json')
        previous = None
        if marker.is_file():
            try:
                previous = json.loads(marker.read_text())
            except ValueError:
                previous = None
            if previous and previous.get('archive') == name and all(
                    (DEST / rel).is_file() and (DEST / rel).stat().st_size == size
                    for rel, size in previous.get('kept', [])):
                if previous.get('kept'):
                    skipped_resume += 1
                    print('skip {}: scanned before, {} clip(s) still on disk'
                          .format(name, len(previous['kept'])), flush=True)
                else:
                    skipped_empty += 1
                    print('skip {}: indexed before, holds no referenced clip'.format(name), flush=True)
                continue
        streamed += entry['size']
        counter = None
        kept = []
        for attempt in range(4):
            kept = []
            try:
                if shutil.disk_usage(DEST).free < MIN_FREE:
                    raise RuntimeError('less than {:.0f} GiB free on {}'.format(MIN_FREE / 2 ** 30, DEST))
                print('stream {} ({:.2f} GiB) attempt {}'.format(name, entry['size'] / 2 ** 30, attempt + 1),
                      flush=True)
                with open_stream(resolve_url(name)) as raw:
                    counter = CountingReader(raw)
                    with tarfile.open(fileobj=counter, mode='r|gz') as archive:
                        for member in archive:
                            rel = member.name.removeprefix('./')
                            if not member.isfile():
                                continue
                            if rel in wanted_set:
                                pass
                            else:
                                # repo dir is first_segment_video/, JSON paths are
                                # first_part_video/..., so match the basename too and
                                # keep only unambiguous names.
                                same = by_name.get(Path(rel).name, [])
                                if len(same) != 1:
                                    continue
                                rel = same[0]
                            target = DEST / rel
                            if target.is_file() and target.stat().st_size == member.size:
                                kept.append((rel, member.size))
                                pending.discard(rel)
                                continue
                            # verify the member size gate before every write
                            if shutil.disk_usage(DEST).free < member.size + MIN_FREE:
                                raise RuntimeError('insufficient disk for {}'.format(rel))
                            target.parent.mkdir(parents=True, exist_ok=True)
                            # with_name (not with_suffix) keeps multi-dot clip names intact
                            partial = target.with_name(target.name + '.partial')
                            with archive.extractfile(member) as source, partial.open('wb') as out:
                                shutil.copyfileobj(source, out, length=1024 * 1024)
                            if partial.stat().st_size != member.size:
                                raise RuntimeError('{}: short write'.format(rel))
                            partial.replace(target)
                            kept.append((rel, member.size))
                            pending.discard(rel)
                            if not pending:
                                break
                save(marker, {'archive': name, 'bytes': entry['size'], 'complete': True,
                              'kept': kept, 'n_kept': len(kept)})
                bytes_read += counter.n
                break
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError('failed archive {}: {}'.format(name, type(exc).__name__)) from None
                print('  retry {}: {}'.format(name, type(exc).__name__), flush=True)
                time.sleep(5)
        scanned += 1
        with_needs += 1 if kept else 0
        print('  kept {} referenced clip(s), {} bytes read{}'
              .format(len(kept), counter.n, '' if kept else '  (no referenced clip here)'), flush=True)

    missing = sorted(rel for rel in distinct if not (DEST / rel).is_file())
    if missing:
        save(DEST / 'FAILED.json', {'missing': missing[:50], 'n_missing': len(missing),
                                    'note': 'Do not evaluate a partial benchmark.'})
        raise SystemExit('{} referenced clip(s) missing after scanning all archives'.format(len(missing)))

    for row in rows:
        row['video_path'] = str(DEST / row['video_path'].removeprefix('./'))
    save(DEST / 'futurebench_local.json', rows)

    retained = sum((DEST / rel).stat().st_size for rel in distinct)
    try:
        import decord
    except ImportError:
        decord = None
        print('decord unavailable: skipped the decode check (the eval needs decodable clips)')
    if decord is not None:
        for rel in distinct:
            reader = decord.VideoReader(str(DEST / rel), num_threads=1)
            if len(reader) == 0:
                raise SystemExit('undecodable clip: {}'.format(rel))
            reader.get_batch([0, len(reader) - 1])
        print('decode-checked {} clips'.format(len(distinct)))

    save(DEST / 'futurebench_ready.json', {
        'repo': REPO, 'revision': REV, 'n_questions': len(rows), 'n_distinct_videos': len(distinct),
        'json': str(DEST / 'futurebench_local.json'), 'retained_bytes': retained,
        'network_bytes_read': bytes_read, 'network_bytes_if_unindexed': streamed,
        'archives_scanned': scanned, 'archives_with_referenced_clips': with_needs,
        'archives_skipped_on_resume': skipped_resume, 'archives_skipped_empty': skipped_empty})
    print('futurebench ready: {} clips, {:.2f} GiB on disk, eval json {}'
          .format(len(distinct), retained / 2 ** 30, DEST / 'futurebench_local.json'))
    print('network: {:.1f} GiB read over {} archive(s); skipped {} empty + {} already-indexed'
          .format(bytes_read / 2 ** 30, scanned, skipped_empty, skipped_resume))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        save(DEST / 'FAILED.json', {'error_type': type(exc).__name__, 'message': str(exc),
                                    'note': 'Fetch failed; do not evaluate a partial benchmark.'})
        raise SystemExit('FutureBench fetch failed: {}: {}'.format(type(exc).__name__, exc))
PYEOF
