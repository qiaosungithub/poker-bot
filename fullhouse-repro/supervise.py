"""Durable local training + checkpoint evaluation, with a separate held-out test."""
from pathlib import Path
import os, sys, json, subprocess, time, hashlib, statistics, datetime
ROOT = Path(__file__).resolve().parent
from runtime_paths import TRAIN_PY, EVAL_PY


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, obj):
    tmp = Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def command(argv, logfile):
    with open(logfile, 'a') as f:
        subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def paired(candidate, reference):
    def load(p):
        return {r['seed']: r['scores']['fullhouse'] for r in map(json.loads, Path(p).with_suffix('.jsonl').read_text().splitlines())}
    c, r = load(candidate), load(reference)
    assert c.keys() == r.keys()
    diffs = [c[k]-r[k] for k in c]
    mean = statistics.mean(diffs); se = statistics.stdev(diffs)/len(diffs)**.5
    return dict(mean=mean, se=se, ci95=[mean-1.96*se, mean+1.96*se], blocks=len(diffs))


def evaluate(checkpoint, split):
    ckpt = Path(checkpoint); outdir = ROOT/'eval'; outdir.mkdir(exist_ok=True)
    onnx = outdir/(ckpt.stem+'.onnx')
    if not Path(str(onnx)+'.validation.json').exists():
        command([TRAIN_PY, 'export_validate.py', str(ckpt), str(onnx)], outdir/(ckpt.stem+'_export.log'))
    output = outdir/(ckpt.stem+'_'+split+'.json')
    seed, blocks = (11300000, 256) if split == 'selection' else (12300000, 1024)
    ref = outdir/('reference_'+split+'.json')
    if not ref.exists():
        command([EVAL_PY,'evaluate.py','--output',str(ref),'--seed',str(seed),'--blocks',str(blocks)],ref.with_suffix('.log'))
    if not output.exists():
        command([EVAL_PY,'evaluate.py','--onnx',str(onnx),'--output',str(output),'--seed',str(seed),'--blocks',str(blocks)],output.with_suffix('.log'))
    data = json.loads(output.read_text())
    record = dict(time=stamp(), checkpoint=str(ckpt), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  split=split, score=data['results']['fullhouse'], difference=paired(output, ref), artifact=str(output))
    with open(ROOT/'results.jsonl','a') as f:
        f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
    return record


def main():
    config_path = ROOT/'config_main.json'
    config = json.loads(config_path.read_text())
    run = ROOT/config['checkpoint_dir']
    if (run/'latest.pt').exists():
        config['resume'] = True
        config_path = ROOT/'config_resume.json'
        atomic_json(config_path, config)
    seen = set()
    selected = []
    if (ROOT/'results.jsonl').exists():
        for line in (ROOT/'results.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['split'] == 'selection' and str(run)+'_adv/' in r['checkpoint']:
                seen.add(r['checkpoint']); selected.append(r)
    env = os.environ.copy(); env['JAX_PLATFORMS'] = 'tpu'
    with open(ROOT/'train.log', 'a') as log:
        proc = subprocess.Popen([TRAIN_PY,'-u','launch.py',str(config_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(ROOT/'process.json', dict(supervisor_pid=os.getpid(), trainer_pid=proc.pid, started=stamp(), config=str(config_path)))
    while True:
        candidates = sorted(Path(str(run)+'_adv').glob('iter_*.npz'))
        for ckpt in candidates:
            if str(ckpt) in seen or time.time()-ckpt.stat().st_mtime < 5:
                continue
            try:
                record = evaluate(ckpt, 'selection')
            except Exception as e:
                atomic_json(ROOT/'status.json',dict(time=stamp(),state='evaluation_failed',error=repr(e),checkpoint=str(ckpt),trainer_pid=proc.pid))
                raise
            seen.add(str(ckpt)); selected.append(record)
        rc = proc.poll()
        atomic_json(ROOT/'status.json',dict(time=stamp(),state='training' if rc is None else 'training_ended',returncode=rc,
                    trainer_pid=proc.pid,evaluated_checkpoints=len(seen),latest_result=selected[-1] if selected else None))
        if rc is not None:
            break
        time.sleep(30)
    if rc != 0:
        raise RuntimeError(f'Trainer exited {rc}; see train.log')
    for ckpt in sorted(Path(str(run)+'_adv').glob('iter_*.npz')):
        if str(ckpt) not in seen:
            selected.append(evaluate(ckpt,'selection'))
    best = max(selected, key=lambda r:r['score']['mean'])
    final = str(Path(str(run)+'_adv')/f"iter_{config['n_iterations']:05d}.npz")
    confirmation = {label:evaluate(path,'confirmation') for label,path in [('selected',best['checkpoint']),('final_budget',final)]}
    passed = confirmation['selected']['difference']['ci95'][0] > -2
    atomic_json(ROOT/'status.json',dict(time=stamp(),state='complete' if passed else 'needs_further_training',
        criterion_passed=passed,confirmation=confirmation,selection=best))


if __name__ == '__main__':
    main()
