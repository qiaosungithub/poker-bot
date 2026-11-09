"""Build the reviewed source-only submission with the official pack checker."""
import hashlib,json,shutil
from pathlib import Path
from dataclasses import asdict
from poker_agent_sdk.bundle import pack, check_submission
root=Path(__file__).resolve().parent
stage=root/'.package';stage.mkdir(exist_ok=True)
for name in ('agent.py','auction.py'):shutil.copyfile(root/name,stage/name)
(root/'dist').mkdir(exist_ok=True)
fingerprint=hashlib.sha256(b''.join((root/n).read_bytes() for n in ('agent.py','auction.py'))).hexdigest()[:12]
archive=root/'dist'/f'variant-v1-{fingerprint}.zip'
report=check_submission(archive.read_bytes()) if archive.exists() else pack(stage,archive)
manifest={'path':str(archive),'bundle':asdict(report),'source_hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('agent.py','auction.py')}}
(root/'evidence/package.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
