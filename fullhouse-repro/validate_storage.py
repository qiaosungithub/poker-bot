"""Exercise relocated Python environments and the actual model serving path."""
import runtime_paths
from pathlib import Path
import os, sys, json, subprocess, hashlib

ROOT = Path(__file__).resolve().parent


def main():
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['JAX_PLATFORMS'] = 'cpu'
    facts = {}
    for name in ('poker-fullhouse-train', 'poker-fullhouse-train312',
                 'poker-fullhouse-venv', 'poker-hu-validation'):
        py = runtime_paths.ENVIRONMENTS / name / 'bin/python'
        code = 'import sys,json,numpy;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,numpy=numpy.__version__)))'
        facts[name] = json.loads(subprocess.check_output([str(py), '-c', code], cwd=ROOT, env=env, text=True))
        assert str(runtime_paths.RUNTIME) in facts[name]['prefix']
    code = ('import runtime_paths,tempfile,torch,jax,onnxruntime; '
            'assert torch.ones(2,2).matmul(torch.ones(2,2)).sum().item()==8; '
            'assert str(runtime_paths.RUNTIME) in tempfile.gettempdir(); '
            'print(torch.__version__,jax.__version__,onnxruntime.__version__,tempfile.gettempdir())')
    facts['training_imports'] = subprocess.check_output([runtime_paths.TRAIN_PY,'-c',code],cwd=ROOT,env=env,text=True).strip()
    out = runtime_paths.RUNTIME/'verification'; out.mkdir(exist_ok=True)
    ckpt = ROOT/'runs/main_adv/iter_00210.npz'
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert digest == json.loads((ROOT/'status.json').read_text())['confirmation']['selected']['checkpoint_sha256']
    with open(out/'export.log','w') as log:
        subprocess.run([runtime_paths.TRAIN_PY,'export_validate.py',str(ckpt),str(out/'model.onnx')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    facts['export'] = json.loads((out/'model.onnx.validation.json').read_text())
    with open(out/'arena.log','w') as log:
        subprocess.run([runtime_paths.EVAL_PY,'evaluate.py','--onnx',str(out/'model.onnx'),'--output',str(out/'arena.json'),
                        '--blocks','8','--workers','4','--seed','10300000'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    arena = json.loads((out/'arena.json').read_text()); assert arena['errors'] == 0 and arena['matches'] == 48
    facts['arena'] = dict(errors=arena['errors'],matches=arena['matches'])
    facts['selected_checkpoint_sha256'] = digest
    facts['passed'] = True
    (ROOT/'storage_relocation_validation.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(facts,indent=2))


if __name__ == '__main__':
    main()
