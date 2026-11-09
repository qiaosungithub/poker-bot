"""Load pinned policies with isolated memory and explicit sampling-only edits."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import random
import sys
import types

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'hu-benchmark'), str(ROOT/'latency-survey')]
import multiway_benchmark as engine

REPO = ROOT/'latency-survey/fullhouse-bot'
SOURCES = {
    'solver_like': REPO/'bots/adversarial/solver_like.py',
    'tournament_specialist': REPO/'bots/adversarial/tournament_specialist.py',
    'strong': REPO/'bots/strong/bot.py',
    'cfr_lite': REPO/'bots/competitors/cfr_lite/bot.py',
    'hydra': REPO/'bots/competitors/adaptive_hydra/bot.py',
    'txnujk': ROOT/'txnujk-bot/bots/floorfix/bot.py',
    'vikram': ROOT/'external/fullhouse-vikram/bots/submission_candidate/bot.py',
    'pantheon': ROOT/'external/fullhouse-lucas/hero_bots/bot_v16_squeeze_C_metacap/bot.py',
    'fold_equity': ROOT/'external/fold-equity/bot.py',
    'station': REPO/'bots/adversarial/station.py',
    'maniac': REPO/'bots/adversarial/maniac.py',
}
CANDIDATES = ['solver_like','tournament_specialist','strong'] + [
    f'{base}_{variant}' for base in ['cfr_lite','hydra','txnujk','vikram','pantheon','fold_equity']
    for variant in ['original','32','64']]
CONTROLS = ['fullhouse_released','solver_like','hydra_32','station','maniac']
_CODE = {}
_PREFLOP = None
_FULLHOUSE = None
_ONNX = {}


def spec(name):
    if name in SOURCES:
        return name, None
    for suffix in ('_original','_32','_64'):
        if name.endswith(suffix) and name[:-len(suffix)] in SOURCES:
            return name[:-len(suffix)], None if suffix=='_original' else int(suffix[1:])
    raise KeyError(name)


class Diagnostics(ast.NodeTransformer):
    """Observe swallowed exceptions; successful decision paths are unchanged."""
    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        node.body.insert(0, ast.Expr(ast.Call(ast.Name('_record_exception',ast.Load()),[],[])))
        return node


def source_code(name):
    if name in _CODE:
        return _CODE[name]
    base, cap = spec(name)
    path = SOURCES[base]
    tree = ast.parse(path.read_text())
    edits = []
    if cap and base=='pantheon':
        for fn in ast.walk(tree):
            if isinstance(fn,ast.FunctionDef) and fn.name in ('_equity_vs_one_range','_equity_vs_multi_range'):
                arguments=fn.args.args[-len(fn.args.defaults):]
                for i,arg in enumerate(arguments):
                    if arg.arg=='max_iters':
                        edits.append({'function':fn.name,'assignment':'max_iters default','before':ast.unparse(fn.args.defaults[i]),'after':str(cap)})
                        fn.args.defaults[i]=ast.Constant(cap)
        assert len(edits)==2
    if cap and base in ('hydra','vikram','txnujk'):
        function = {'hydra':'_mc_equity','vikram':'_monte_carlo','txnujk':'_postflop'}[base]
        target = 'samples' if base=='txnujk' else 'max_iters'
        for fn in ast.walk(tree):
            if isinstance(fn,ast.FunctionDef) and fn.name==function:
                for node in ast.walk(fn):
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in node.targets):
                        edits.append({'function':function,'assignment':target,'before':ast.unparse(node.value),'after':str(cap)})
                        node.value=ast.Constant(cap)
        assert len(edits)==(3 if base=='vikram' else 1),(name,edits)
    # Cache only txnujk's deterministic import-time equity table, never policy state.
    if base=='txnujk':
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFLOP_EQ' for t in node.targets) and isinstance(node.value,ast.Call):
                node.value=ast.IfExp(ast.Compare(ast.Name('_cached_preflop',ast.Load()),[ast.IsNot()],[ast.Constant(None)]),ast.Name('_cached_preflop',ast.Load()),node.value)
    diagnostic_tree=ast.fix_missing_locations(Diagnostics().visit(tree))
    code=compile(diagnostic_tree,str(path),'exec')
    source=ast.unparse(diagnostic_tree)
    meta={'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'instrumented_code_sha256':hashlib.sha256(source.encode()).hexdigest(),'sampling_cap':cap,'ast_sampling_edits':edits}
    _CODE[name]=(code,meta)
    return code,meta


def initialize_fullhouse():
    global _FULLHOUSE
    if _FULLHOUSE is None:
        engine.C.R.initialize()
        _FULLHOUSE=engine.C.R.HERO
    return _FULLHOUSE


class Policy:
    def __init__(self,name,seed):
        global _PREFLOP
        self.name=name
        self.exceptions=[]
        self.active=False
        self.random_state=random.Random(seed).getstate()
        if name.startswith('fullhouse_'):
            base=initialize_fullhouse()
            namespace=dict(vars(base))
            # Rebind all module-defined functions to the private module namespace.
            for key,value in list(namespace.items()):
                if isinstance(value,types.FunctionType) and value.__globals__ is vars(base):
                    fn=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
                    fn.__kwdefaults__=value.__kwdefaults__
                    namespace[key]=fn
            from fullhouse_onnx import ONNXLookup
            path=ROOT/('latency-survey/fullhouse_fp32.onnx' if name=='fullhouse_released' else 'fullhouse-repro/eval/iter_00210.onnx')
            if name not in _ONNX:_ONNX[name]=ONNXLookup(path)
            namespace['_DEEP_CFR']=_ONNX[name]
            namespace['_RNG']=random.Random(seed)
            self.namespace=namespace
            self.fn=namespace['decide']
        else:
            base,cap=spec(name)
            code,_=source_code(name)
            namespace={'__name__':'handwritten_'+name,'__file__':str(SOURCES[base]),
                       '_record_exception':self.record_exception,'_cached_preflop':_PREFLOP}
            old=random.getstate();random.seed(198011)
            try:exec(code,namespace)
            finally:random.setstate(old)
            if base=='txnujk':
                assert len(namespace['PREFLOP_EQ'])==169
                _PREFLOP=namespace['PREFLOP_EQ']
            if cap and base=='cfr_lite':
                cfg=copy.deepcopy(namespace['CONFIG'])
                cfg['equity_iterations'].update(flop=cap,turn=cap,river=cap,heads_up_bonus=0,multiway_penalty=0,multiway_floor=cap)
                namespace['CONFIG']=cfg
            if cap and base=='pantheon':
                namespace['_EQ_MIN_ITERS']=cap
                namespace['_EQ_MAX_ITERS']=cap
                namespace['_EQ_STREET_BUDGET']={k:(v[0],cap) for k,v in namespace['_EQ_STREET_BUDGET'].items()}
            if cap and base=='fold_equity':
                namespace['_MC_OPP_HANDS']=cap
                namespace['_MC_RUNOUTS_PER_HAND']=1
            for value in namespace.values():
                if isinstance(value,random.Random):value.seed(seed)
            self.namespace=namespace
            self.fn=namespace['decide']

    def record_exception(self):
        if self.active:
            ex=sys.exc_info()[1]
            self.exceptions.append(type(ex).__name__+': '+str(ex)[:160])

    def enter(self):
        random.setstate(self.random_state)
        self.active=True

    def leave(self):
        self.active=False
        self.random_state=random.getstate()

    def act(self,state):
        return self.fn(state)


def metadata(name):
    if name.startswith('fullhouse_'):
        path=ROOT/('latency-survey/fullhouse_fp32.onnx' if name=='fullhouse_released' else 'fullhouse-repro/eval/iter_00210.onnx')
        return {'family':name,'learned':True,'model':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    base,cap=spec(name)
    _,meta=source_code(name)
    meta=dict(meta,family=base,learned=False)
    if cap and base=='cfr_lite':meta['runtime_edits']={'CONFIG.equity_iterations':{'flop':cap,'turn':cap,'river':cap,'heads_up_bonus':0,'multiway_penalty':0,'multiway_floor':cap}}
    if cap and base=='pantheon':meta['runtime_edits']={'_EQ_MIN_ITERS':cap,'_EQ_MAX_ITERS':cap,'_EQ_STREET_BUDGET':'keep original time limits; all iteration limits = '+str(cap)}
    if cap and base=='fold_equity':meta['runtime_edits']={'_MC_OPP_HANDS':cap,'_MC_RUNOUTS_PER_HAND':1}
    return meta
