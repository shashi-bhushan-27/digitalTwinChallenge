from common.io import content_hash
from data_gen.generate import generate, load_configs


def test_dataset_regenerates_identically(cfgs):
    a = generate(cfgs, n_patients=5, n_days=2, verbose=False)
    b = generate(cfgs, n_patients=5, n_days=2, verbose=False)
    assert a.keys() == b.keys()
    for name in a:
        assert content_hash(a[name]) == content_hash(b[name]), name


def test_different_seed_changes_data():
    cfgs = load_configs(overrides={"pipeline": {"seed": 999}})
    other = generate(cfgs, n_patients=5, n_days=2, verbose=False)
    base = generate(load_configs(), n_patients=5, n_days=2, verbose=False)
    assert content_hash(other["observed/cgm"]) != content_hash(base["observed/cgm"])


def test_observed_tables_hold_no_hidden_state(small_dataset):
    hidden = {"si", "beta", "egp_max", "adherence_true", "stress", "illness", "carbs_g", "plate_id"}
    for name, df in small_dataset.items():
        if name.startswith("observed/"):
            assert not hidden & set(df.columns), name


def test_splits_partition_patients(small_dataset):
    s = small_dataset["observed/splits"]
    assert s["patient_id"].is_unique
    assert set(s["split"]) <= {"train", "val", "test"}
    assert set(s["patient_id"]) == set(small_dataset["observed/patients"]["patient_id"])


def test_observation_window(small_dataset):
    cgm = small_dataset["observed/cgm"]
    days = (cgm["timestamp"].max() - cgm["timestamp"].min()).total_seconds() / 86400
    assert 3.9 < days <= 4.0
