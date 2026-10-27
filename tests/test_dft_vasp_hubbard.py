"""Input identity tests use real ASE writers; no electronic-structure executable."""
from dataclasses import replace
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from NepTrain.core.dft.vasp import native
from NepTrain.core.dft.vasp.io import VaspInput, read_vasp_input
from NepTrain.core.dft.vasp.hubbard import validate_hubbard, configure_hubbard, audit_written_input
from NepTrain.core.dft.vasp.deltaspin import DeltaSpinResult
from NepTrain.core.labeling import label
from test_dft_vasp_deltaspin import _request

HUBBARD = {'Fe': {'L': 2, 'U': 4., 'J': 0.}, 'Bi': {'L': -1, 'U': 0., 'J': 0.}, 'O': {'L': -1, 'U': 0., 'J': 0.}}
INCAR = 'LNONCOLLINEAR=.TRUE.\nLDELTASPIN=.TRUE.\nLDAU=.TRUE.\nLDAUTYPE=2\nKSPACING=0.25\n'


@pytest.mark.parametrize('natoms', [10, 40, 50])
@pytest.mark.parametrize('order', [*permutations(['Fe', 'Bi', 'O']), 'interleaved'])
def test_real_input_generation_and_output_identity(tmp_path, monkeypatch, natoms, order):
    n = natoms // 5
    symbols = ['Fe']*n + ['Bi']*n + ['O']*(3*n)
    if order == 'interleaved':
        np.random.default_rng(918).shuffle(symbols)
    else:
        symbols = [s for element in order for s in symbols if s == element]
    a = Atoms(symbols, positions=np.arange(natoms*3).reshape(-1, 3)*0.03, cell=[100]*3, pbc=True)
    spin = np.column_stack([np.arange(natoms)*0.001, np.zeros(natoms), np.full(natoms, 4.)])
    a.set_array('spin', spin)
    request = _request(tmp_path, a, INCAR)
    request.settings.update(hubbard=HUBBARD, constrained_elements=['Fe'])
    seen = []

    class Replay(VaspInput):
        def _run(self, **kwargs):
            # The new audit must run before launch, not after convergence.
            seen.append(self.input_audit)
            return 0, ''
        def read_results(self):
            self.converged = True
            self.results = {'energy': -1., 'forces': self.atoms.positions.copy(), 'stress': np.zeros(6)}

    def result(path, species, targets, active):
        active = np.asarray(active, dtype=bool)
        force = targets.copy(); force[~active] = 0
        return DeltaSpinResult(targets.copy(), force, 0, 0., 1e-6, active, targets.copy())

    monkeypatch.setattr(native, 'VaspInput', Replay)
    monkeypatch.setattr(native.deltaspin, 'read_result', result)
    output = read(label(request, 'vasp').output_file)
    np.testing.assert_allclose(output.get_forces(), a.positions, atol=1e-8)
    np.testing.assert_allclose(output.arrays['spin'], spin, atol=1e-8)
    mf = spin.copy(); mf[np.array(symbols) != 'Fe'] = 0
    np.testing.assert_allclose(output.arrays['mforce'], mf, atol=1e-8)
    evidence = seen[0]
    actual_species = list(dict.fromkeys(symbols))
    assert evidence['species'] == actual_species
    assert evidence['expanded_hubbard']['LDAUU'] == [HUBBARD[s]['U'] for s in actual_species]
    order_map = np.asarray(evidence['vasp_to_original'])
    inverse = np.asarray(evidence['original_to_vasp'])
    np.testing.assert_array_equal(order_map[inverse], np.arange(natoms))
    manifest = json.loads(next(request.work_dir.glob('*/vasp-result.json')).read_text())
    assert manifest['input_binding'] == evidence
    assert evidence['input_sha256'] == manifest['input_sha256']


@pytest.mark.parametrize('bad', [
    {}, {'Fex': {'L': 2, 'U': 4., 'J': 0.}}, {'Fe': {'L': 2, 'U': 4.}},
    {'Fe': {'L': True, 'U': 4., 'J': 0.}}, {'Fe': {'L': 4, 'U': 4., 'J': 0.}},
    {'Fe': {'L': 2, 'U': float('nan'), 'J': 0.}},
    {'Fe': {'L': 2, 'U': True, 'J': 0.}}, {'Fe': {'L': -1, 'U': 4., 'J': 0.}},
    {'Fe': {'L': 2, 'U': 4.0001, 'J': 0.}},
])
def test_invalid_hubbard_is_rejected(bad):
    with pytest.raises(ValueError): validate_hubbard(bad)


@pytest.mark.parametrize('extra', ['LDAUL=2 -1 -1', 'LDAUU=4 0 0', 'LDAUJ=0 0 0', 'LDAUTYPE=1', 'LDAU=.FALSE.'])
def test_conflicting_incar_rejected_before_execution(tmp_path, extra):
    path = tmp_path/'INCAR';path.write_text(INCAR+'\n'+extra+'\n')
    with pytest.raises(native.NativeVaspError):
        native.validate_vasp_input_file(path, hubbard=HUBBARD)


def test_every_present_species_must_be_explicit():
    with pytest.raises(ValueError, match='missing Hubbard elements'):
        validate_hubbard({'Fe': HUBBARD['Fe']}, ['Fe', 'Bi', 'O'])


def test_written_files_are_checked_before_launch(tmp_path, monkeypatch):
    # Reuse the real-input oracle, then mutate its generated artifacts individually.
    from ase.calculators.vasp import Vasp
    from NepTrain.core.dft.vasp.deltaspin import prepare_input
    from NepTrain.core.dft.vasp.resources import validate_vasp_resources
    a=Atoms('FeBiFeO6Bi',positions=np.arange(30).reshape(-1,3)*.1,cell=[10]*3,pbc=True)
    a.set_array('spin', np.tile([0.,0.,4.],(len(a),1)))
    a.set_array('spin_active', np.array([int(s=='Fe') for s in a.symbols]))
    request=_request(tmp_path,a,INCAR);d=tmp_path/'written';d.mkdir()
    c=VaspInput(pp_path=request.settings['resource_dir'],directory=str(d));read_vasp_input(c,request.settings['input_file'])
    configure_hubbard(c,HUBBARD,a.symbols)
    resources=validate_vasp_resources(request.settings['resource_dir'],request.settings['resource_manifest'],a)
    c.set(pp='',setups={'base':'minimal',**resources['ase_setups']})
    b,resort=prepare_input(c,a)
    c.input_binding=dict(hubbard=HUBBARD,order=np.argsort(resort).tolist(),potcar_sha256=resources['combined_sha256'],deltaspin=True,active_mask=b.arrays['spin_active'].tolist())
    c.write_input(b)
    saved={name:(d/name).read_text() for name in ['INCAR','POTCAR','POSCAR']}
    for name, content in [
        ('POTCAR', saved['POTCAR']+'corruption'),
        ('INCAR', saved['INCAR'].replace('4.000', '5.000')),
        ('INCAR', saved['INCAR']+'\nM_DELTASPIN = 0 0 4\n'),
        ('INCAR', saved['INCAR']+'\nDELTASPIN_ATOMS = 10*1\n'),
    ]:
        (d/name).write_text(content)
        with pytest.raises(ValueError): audit_written_input(c,b)
        (d/name).write_text(saved[name])


def test_manual_prepare_carries_hubbard_and_checks_species(tmp_path):
    from NepTrain.core.manual import prepare_labeling, ManualTaskError
    from NepTrain.core.execution import ExecutionTarget
    a=Atoms('FeBiO3',positions=np.arange(15).reshape(-1,3)*.1,cell=[10]*3,pbc=True)
    a.set_array('spin',np.tile([0.,0.,4.],(5,1)))
    request=_request(tmp_path,a,INCAR)
    kwargs=dict(backend='vasp',output=tmp_path/'manual-output.xyz',workdir=tmp_path/'manual',
                target=ExecutionTarget('local','process'),input_file=request.settings['input_file'],
                resource_dir=request.settings['resource_dir'],resource_manifest=request.settings['resource_manifest'],
                constrained_elements=['Fe'],hubbard=HUBBARD)
    prepare_labeling(request.source,**kwargs)
    requests=list((tmp_path/'manual').rglob('request.json'))
    assert requests and json.loads(requests[0].read_text())['hubbard']==HUBBARD
    kwargs.update(workdir=tmp_path/'missing',hubbard={'Fe':HUBBARD['Fe']})
    with pytest.raises(ManualTaskError,match='missing Hubbard'):
        prepare_labeling(request.source,**kwargs)


def test_config_rejects_hubbard_for_other_backends(tmp_path):
    from test_config import _project, _write
    from NepTrain.core.config import load_config, ConfigError
    load_config(_write(tmp_path,_project(labeling={'backend':'vasp','hubbard':HUBBARD})))
    with pytest.raises(ConfigError,match='backend=vasp'):
        load_config(_write(tmp_path,_project(labeling={'backend':'abacus','hubbard':HUBBARD})))


def test_workflow_adopts_first_label_mapping_without_retraining(tmp_path):
    from test_workflow import _inputs
    from NepTrain.core.config import load_config, save_config
    from NepTrain.core.workflow import prepare_workflow, WorkflowError
    from NepTrain.core.workflow_config import accept_config_changes, WorkflowConfigError
    from ase.io import write
    path,initial=_inputs(tmp_path)
    # Real resource validation and workflow preparation, no scheduler submission.
    resources_dir=tmp_path/'resources-test';resources_dir.mkdir()
    a=Atoms('Fe',positions=[[0,0,0]],cell=[4]*3,pbc=True)
    req=_request(resources_dir,a,'LDAU=.TRUE.\nLDAUL=2\nLDAUU=4\nLDAUJ=0\n')
    write(tmp_path/'structure.xyz',a,format='extxyz')
    config,_=load_config(path)
    config['labeling']=dict(backend='vasp',input_path=str(req.settings['input_file']),
                           resource_path=str(req.settings['resource_dir']),potcar_manifest_path=str(req.settings['resource_manifest']))
    save_config(config,path)
    prep=prepare_workflow(path,initial,tmp_path/'workflow')
    root=prep.output_dir;state=root/'.neptrain';(state/'jobs').mkdir(exist_ok=True)
    (state/'controller.json').write_text(json.dumps({'state':'stopped','current':{'stage':'train','handle':{'execution_id':'keep-me'}},'history':[]}))
    (state/'ledger.json').write_text(json.dumps({'generations':{'1':{'stages':{}}}}))
    job=state/'jobs/g0001-train-a1-existing';job.mkdir();(job/'task.json').write_text('{"stage":"train"}')
    controller_before=(state/'controller.json').read_bytes()
    manifest_before=json.loads(prep.manifest.read_text())
    new_input=root/'inputs/labeling/hubbard.INCAR';new_input.write_text('LDAU=.TRUE.\nLDAUTYPE=2\n')
    config,_=load_config(prep.config_file);config['labeling']['hubbard']={'Fe':dict(HUBBARD['Fe'])};config['labeling']['input_path']=str(new_input);save_config(config,prep.config_file)
    changes=accept_config_changes(root)
    assert any(c['field'].startswith('labeling.hubbard') for c in changes)
    after=json.loads(prep.manifest.read_text())
    assert after['initial_training']==manifest_before['initial_training']
    assert after['plans']==manifest_before['plans']
    assert (state/'controller.json').read_bytes()==controller_before
    assert any(r['path']==str(new_input) for r in after['dependencies'] if r['role']=='labeling_input')
    assert not accept_config_changes(root)
    # Even a prepared-but-not-submitted label bundle seals the physics.
    (state/'jobs/g0001-label-a1-prepared').mkdir()
    config['labeling']['hubbard']['Fe']['U']=5.;save_config(config,prep.config_file)
    with pytest.raises(WorkflowError,match='cannot be applied'):
        accept_config_changes(root)
