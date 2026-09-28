import pytest

from common import config
from data_gen.generate import generate, load_configs


@pytest.fixture(scope="session")
def cfgs():
    return load_configs()


@pytest.fixture(scope="session")
def small_dataset(cfgs):
    """A small end-to-end dataset shared by the layer tests (12 patients x 4 days)."""
    return generate(cfgs, n_patients=12, n_days=4, verbose=False)


@pytest.fixture(scope="session")
def cohort_cfg():
    return config.load("cohort")
