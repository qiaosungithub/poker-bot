"""Compose source-only poker bots from explicit frozen module revisions.

This is a development utility, never part of a submitted package. It refuses
existing output paths and records every original/generated source hash.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

EXPORTS = {'auction_policy.py': 'choose_bid', 'betting_policy.py': 'choose_bet',
           'preflop_policy.py': 'choose_preflop', 'postflop_policy.py': 'choose_postflop',
           'flop_policy.py': 'choose_flop', 'turn_policy.py': 'choose_turn',
           'river_policy.py': 'choose_river'}


def compose(config, output, repo):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    def revision(ref):
        if not isinstance(ref, str) or ref.startswith('-'):
            raise ValueError('Invalid source revision')
        return git('rev-parse', '--verify', ref + '^{commit}').decode().strip()
    def safe_path(path):
        value = PurePosixPath(path)
        if value.is_absolute() or '..' in value.parts or value.suffix != '.py':
            raise ValueError('Module paths must be relative Python source paths')
        return str(value)
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    output = output.resolve()
    manifest_path = output.with_name(output.name + '.manifest.json')
    if output.exists() or manifest_path.exists():
        raise FileExistsError('Use a new output directory and manifest path')
    base = revision(config.get('base_ref', 'modules-five-stage-v2'))
    files, sources = {}, {}
    tree = git('ls-tree', '-r', '-z', base, '--', 'bot/')
    for record in tree.split(b'\0'):
        if not record:
            continue
        header, path = record.split(b'\t', 1)
        mode, kind, object_id = header.split()
        if kind != b'blob' or mode not in (b'100644', b'100755'):
            raise ValueError('Base bot must contain only ordinary source files')
        source = path.decode()
        target = safe_path(source.removeprefix('bot/'))
        data = git('cat-file', 'blob', object_id.decode())
        files[target] = data
        sources[target] = {'commit': base, 'source': source, 'sha256': digest(data)}
    for item in config.get('replacements', []):
        source, target = safe_path(item['source']), safe_path(item.get('target', item['source']))
        commit = revision(item['commit'])
        data = git('show', commit + ':' + source)
        files[target] = data
        sources[target] = {'commit': commit, 'source': source, 'sha256': digest(data)}
    for item in config.get('bindings', []):
        target = safe_path(item['target'])
        module, function = item['module'], item['function']
        export = item.get('export', EXPORTS.get(target))
        if any(not isinstance(v, str) or not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', v)
               for v in (module, function, export)):
            raise ValueError('Binding names must be plain Python identifiers')
        if module + '.py' not in files:
            raise ValueError('Binding dependency was not included: ' + module)
        data = ('"""Generated frozen policy binding; see external source manifest."""\n'
                f'from {module} import {function} as {export}\n').encode()
        files[target] = data
        sources[target] = {'binding': item, 'sha256': digest(data)}
    if 'agent.py' not in files:
        raise ValueError('No Agent entrypoint in composition')
    output.mkdir(parents=True)
    for target, data in sorted(files.items()):
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    manifest = {'base_commit': base, 'configuration': config, 'sources': sources}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--pack', action='store_true')
    args = parser.parse_args()
    if args.pack and Path(str(args.output) + '.zip').exists():
        parser.error('Package output already exists')
    manifest_path, manifest = compose(json.loads(args.config.read_text()), args.output, args.repo)
    print(json.dumps({'output': str(args.output.resolve()), 'manifest': str(manifest_path),
                      'files': len(manifest['sources']), 'base_commit': manifest['base_commit']}))
    if args.pack:
        subprocess.run([str(Path(sys.executable).with_name('poker-agent')), 'pack',
                        str(args.output), str(args.output) + '.zip'], check=True)


if __name__ == '__main__':
    main()
