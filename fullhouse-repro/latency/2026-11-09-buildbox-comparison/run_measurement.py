"""Run the unchanged latency benchmark and retain host/load provenance."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time
import urllib.request


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_times(cpu):
    siblings = read_optional(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
    ids = {cpu}
    if siblings:
        for part in siblings.split(','):
            bounds = list(map(int, part.split('-')))
            ids.update(range(bounds[0], bounds[-1] + 1))
    names = {'cpu'} | {f'cpu{i}' for i in ids}
    return {parts[0]: list(map(int, parts[1:]))
            for line in Path('/proc/stat').read_text().splitlines()
            if (parts := line.split())[0] in names}


def instance_type():
    # Read this host's non-secret instance type only; never request IAM data.
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            'http://169.254.169.254/latest/api/token', method='PUT',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
        with opener.open(request, timeout=1) as response:
            token = response.read().decode()
        request = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/instance-type',
            headers={'X-aws-ec2-metadata-token': token})
        with opener.open(request, timeout=1) as response:
            return response.read().decode()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    assert not output.exists(), output
    assert args.cpu in os.sched_getaffinity(0)
    manifest = json.loads((root / 'source-manifest.json').read_text())
    for name, expected in manifest.items():
        raw = (root / name).read_bytes()
        assert len(raw) == expected['bytes'], name
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], name
    output.parent.mkdir(parents=True, exist_ok=True)
    host = {
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'hostname': platform.node(), 'platform': platform.platform(),
        'python': sys.version, 'python_config_args': sysconfig.get_config_var('CONFIG_ARGS'),
        'logical_cpus': os.cpu_count(), 'allowed_cpus': sorted(os.sched_getaffinity(0)),
        'selected_cpu': args.cpu,
        'selected_cpu_siblings': read_optional(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list'),
        'lscpu': json.loads(subprocess.check_output(['lscpu', '-J'], text=True)),
        'memory': Path('/proc/meminfo').read_text().splitlines()[:3],
        'ec2_instance_type': instance_type() if platform.node() == 'boston-team-95' else None,
        'cgroup_membership': read_optional('/proc/self/cgroup'),
        'cgroup_cpu_max': read_optional('/sys/fs/cgroup/cpu.max'),
        'cgroup_cpu_stat_before': read_optional('/sys/fs/cgroup/cpu.stat'),
        'load_before': os.getloadavg(), 'cpu_times_before': cpu_times(args.cpu),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'onnxruntime', 'eval7', 'pokerkit')},
        'manifest': manifest,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    env = dict(os.environ)
    env.pop('LD_PRELOAD', None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    command = [sys.executable, str(root / 'fullhouse-repro/measure_latency.py'),
               '--output-dir', str(output), '--cpu', str(args.cpu), '--repeats', '3']
    started = time.perf_counter()
    with output.with_suffix('.log').open('x') as log:
        completed = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    host.update(finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                elapsed_with_startup_seconds=time.perf_counter() - started,
                load_after=os.getloadavg(), cpu_times_after=cpu_times(args.cpu),
                cgroup_cpu_stat_after=read_optional('/sys/fs/cgroup/cpu.stat'),
                benchmark_exit_code=completed.returncode)
    if not output.exists():
        raise RuntimeError(f'Benchmark did not create output; inspect {output.with_suffix(".log")}')
    (output / 'host.json').write_text(json.dumps(host, indent=2) + '\n')
    completed.check_returncode()
    result = json.loads((output / 'result.json').read_text())
    print(json.dumps({'hostname': host['hostname'], 'python': host['python'],
                     'instance_type': host['ec2_instance_type'],
                     'results': {k: v['wall'] for k, v in result['results'].items()}}), flush=True)


if __name__ == '__main__':
    main()
