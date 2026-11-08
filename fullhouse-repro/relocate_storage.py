"""One-time, manifest-backed relocation of this poker project's local artifacts."""
from pathlib import Path
import datetime, hashlib, json, os, pwd, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / '.runtime'
MANIFEST = ROOT / 'storage_relocation.json'
PAIRS = [(Path('/tmp') / name, RUNTIME / 'envs' / name) for name in (
    'poker-fullhouse-train', 'poker-fullhouse-train312',
    'poker-fullhouse-venv', 'poker-hu-validation')]
PAIRS += [(Path('/tmp') / name, RUNTIME / 'survey' / name) for name in (
    'poker-candidates-existing', 'poker-candidates-fast', 'poker-candidates-nn')]
PAIRS += [
    (Path('/tmp/fullhouse-repro-tpu-logs'), RUNTIME / 'logs' / 'tpu'),
    (Path('/home/sqa/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu'),
     RUNTIME / 'python' / 'cpython-3.12.16-linux-x86_64-gnu'),
]


def persist(m):
    p = MANIFEST.with_suffix('.json.tmp')
    p.write_text(json.dumps(m, indent=2)+'\n'); p.replace(MANIFEST)


def active_references():
    prefixes = [str(src) for src, _ in PAIRS]
    hits = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'relocate_storage.py' in args:
                continue
            values = [args, (p/'maps').read_text(errors='replace')]
            for fd in (p/'fd').iterdir():
                try: values.append(os.readlink(fd))
                except OSError: pass
            found = [q for q in prefixes if any(q in v for v in values)]
            if found: hits.append({'pid':int(p.name), 'paths':found})
        except (OSError, PermissionError):
            pass
    return hits


def copy():
    assert not active_references(), 'Source is in use'
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text()); assert m['status'] == 'copying'
        records = m['entries']
        assert [(r['source'],r['destination']) for r in records] == [(str(s),str(d)) for s,d in PAIRS]
    else:
        records = []
        for src, dst in PAIRS:
            assert src.is_dir() and not src.is_symlink(), src
            assert src.stat().st_uid == os.getuid(), src
            assert not dst.exists(), dst
            size = int(subprocess.check_output(['du','-sx','--block-size=1',str(src)],text=True).split()[0])
            records.append(dict(source=str(src), destination=str(dst), allocated_bytes=size,
                                owner=pwd.getpwuid(src.stat().st_uid).pw_name, source_device=src.stat().st_dev))
        m = dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 root_free_before=shutil.disk_usage('/').free, entries=records, status='copying')
    persist(m)
    lock = threading.Lock()
    def one(record):
        if record.get('checksum_copy_verified'):
            return
        src, dst = Path(record['source']), Path(record['destination'])
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync','-aH',str(src)+'/',str(dst)+'/'],check=True)
        verify = subprocess.check_output(['rsync','-aH','--checksum','--dry-run','--itemize-changes',str(src)+'/',str(dst)+'/'],text=True)
        assert not verify.strip(), verify
        with lock:
            record['checksum_copy_verified'] = True
            persist(m)
        print('Copied and checksum verified:',src,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one,records))
    m['status'] = 'copied_and_verified'; persist(m)


def repair():
    m = json.loads(MANIFEST.read_text()); assert m['status'] == 'copied_and_verified'
    replacements = {str(s):str(d) for s,d in PAIRS}
    replacements['/home/sqa/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu'] = str(PAIRS[-1][1])
    modified = []
    for _, dst in PAIRS:
        configs = list(dst.glob('pyvenv.cfg')) + list(dst.glob('venv/pyvenv.cfg'))
        for cfg in configs:
            for p in [cfg] + list((cfg.parent/'bin').iterdir()):
                if p.is_symlink():
                    target = os.readlink(p)
                    for old,new in replacements.items(): target = target.replace(old,new)
                    if target != os.readlink(p):
                        p.unlink(); p.symlink_to(target); modified.append(str(p))
                    continue
                if not p.is_file() or p.stat().st_size > 1024*1024:
                    continue
                raw = p.read_bytes()
                try: text = raw.decode('utf-8')
                except UnicodeDecodeError: continue
                updated = text
                for old,new in replacements.items(): updated = updated.replace(old,new)
                if updated != text:
                    p.write_text(updated); modified.append(str(p))
    m['path_repairs'] = modified; m['status'] = 'paths_repaired'; persist(m)
    print('Repaired environment paths:',len(modified))


def finish():
    m = json.loads(MANIFEST.read_text()); assert m['status'] == 'paths_repaired'
    validation = ROOT/'storage_relocation_validation.json'
    assert json.loads(validation.read_text())['passed'] is True
    hits = active_references(); assert not hits, hits
    # Verify all non-repaired payloads again immediately before source removal.
    repaired = set(m['path_repairs'])
    for record in m['entries']:
        src, dst = Path(record['source']), Path(record['destination'])
        changes = subprocess.check_output(['rsync','-aH','--checksum','--dry-run','--itemize-changes','--out-format=%i|%n',str(src)+'/',str(dst)+'/'],text=True)
        for line in changes.splitlines():
            code, rel = line.split('|',1)
            if code[1] == 'd': continue
            assert str(dst/rel) in repaired, line
        assert src.is_dir() and not src.is_symlink() and src.stat().st_uid == os.getuid()
        shutil.rmtree(src)
        src.symlink_to(dst, target_is_directory=True)
        record['source_replaced_by_compatibility_link'] = True
        persist(m)
        print('Reclaimed root storage:',src,flush=True)
    m['status'] = 'complete'; m['root_free_after'] = shutil.disk_usage('/').free
    m['active_script_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in ROOT.glob('*.py')}
    m['completed'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); persist(m)


if __name__ == '__main__':
    {'copy':copy,'repair':repair,'finish':finish}[sys.argv[1]]()
