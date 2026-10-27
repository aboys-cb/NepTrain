import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from NepTrain.core.scientific_data import (
    INPUT_STRUCTURE_ID_KEY, ScientificDataError,
    known_labeled_structure_ids, structure_id,
)
from NepTrain.core.iteration import StageContext, GenerationPlan
from NepTrain.core.workflow_iteration import WorkflowIterationAdapter, WorkflowIterationError
from NepTrain.core.fps import _nearest_distances, _leave_one_out_nearest_distances, farthest_point_sampling


def frame(energy=-1.0, weight=1.0):
    a = Atoms('FeO', positions=[[0, 0, 0], [2, 0, 0]], cell=[8, 8, 8], pbc=True)
    a.set_array('spin', np.array([[4., 0., 0.], [0., 0., 0.]]))
    a.set_array('mforce', np.zeros((2, 3)))
    a.calc = SinglePointCalculator(a, energy=energy, forces=np.zeros((2, 3)))
    a.info.update(virial=np.zeros((3, 3)), Weight=weight)
    return a


def test_known_inputs_include_constrained_target_and_final_induced_moments():
    source = frame()
    labeled = frame()
    labeled.info[INPUT_STRUCTURE_ID_KEY] = structure_id(source)
    labeled.arrays['spin'][0, 0] += 9e-7
    labeled.arrays['spin'][1, 0] = 0.105
    known = known_labeled_structure_ids([labeled])
    assert structure_id(source) in known
    assert structure_id(labeled) in known
    rotated = frame()
    rotated.arrays['spin'][0] = [0, 4, 0]
    assert structure_id(rotated) not in known  # same geometry, different Fe state
    legacy = frame()
    assert known_labeled_structure_ids([legacy]) == {structure_id(legacy)}
    labeled.info[INPUT_STRUCTURE_ID_KEY] = 'corrupt'
    with pytest.raises(ScientificDataError, match='input_structure_id'):
        known_labeled_structure_ids([labeled])


def merge(tmp_path, old, new):
    write(tmp_path/'old.xyz', old, format='extxyz')
    write(tmp_path/'new.xyz', new, format='extxyz')
    context = StageContext(2, tmp_path, GenerationPlan(2, 1, 100),
                           {'training_input': tmp_path/'old.xyz', 'labeled': tmp_path/'new.xyz'}, {})
    return WorkflowIterationAdapter._merge(None, context)


def test_merge_preserves_consistent_multiplicity_and_weights(tmp_path):
    outcome = merge(tmp_path, [frame(weight=1), frame(weight=3)],
                    [frame(energy=-1+1e-8, weight=7), frame(weight=2)])
    stored = read(outcome.artifacts['training_set'], ':')
    assert len(stored) == 4
    assert [a.info['Weight'] for a in stored] == [1, 3, 7, 2]
    assert outcome.metrics['duplicate_input_count'] == 1
    assert outcome.metrics['duplicate_labeled_count'] == 1
    assert outcome.metrics['duplicate_cross_count'] == 1
    assert outcome.metrics['added_count'] == 2
    assert outcome.metrics['duplicate_policy'] == 'preserve_consistent'


@pytest.mark.parametrize('field', ['energy', 'forces', 'mforce', 'virial'])
def test_merge_rejects_conflicting_labels_across_sets(tmp_path, field):
    other = frame()
    if field == 'energy':
        other.calc.results[field] += 0.01
    elif field == 'forces':
        other.calc.results[field][0, 0] += 0.01
    elif field == 'mforce':
        other.arrays[field][0, 0] += 0.01
    else:
        other.info[field][0, 0] += 0.01
    with pytest.raises(WorkflowIterationError, match='different label sets'):
        merge(tmp_path, [frame()], [other])
    assert not (tmp_path/'train.xyz').exists()


def test_distance_kernel_resolves_identical_and_close_high_dimensional_rows():
    rng = np.random.default_rng(19)
    refs = rng.normal(size=(5, 512))*100
    points = refs.copy()
    points[1, 0] += 1e-8
    distances = _nearest_distances(points, refs, point_batch_size=2, reference_batch_size=2)
    assert distances[0] == 0
    assert distances[2:].tolist() == [0, 0, 0]
    assert distances[1] == pytest.approx(abs(points[1, 0]-refs[1, 0]), rel=1e-12)
    repeated = np.vstack([refs[0], refs[0], points[1], refs[1]])
    loo = _leave_one_out_nearest_distances(repeated, batch_size=2)
    assert loo[:2].tolist() == [0, 0]
    assert loo[2] == pytest.approx(distances[1], rel=1e-12)


def test_uncovered_stratum_cannot_displace_more_novel_spin_state():
    args = dict(reference_descriptors=np.array([[0.]]), candidate_ids=['far', 'new', 'near'],
                strata=['spin', 'spin', 'npt'], min_novelty=1e-9)
    points = np.array([[10.], [8.], [1e-8]])
    result = farthest_point_sampling(points, budget=2, **args)
    assert result.selected_ids == ('far', 'new')
    # Do not impose an arbitrary physical cutoff: a resolved tiny difference
    # remains eligible if the user threshold and budget allow it.
    result = farthest_point_sampling(points, budget=3, **args)
    assert result.selected_ids == ('far', 'new', 'near')
