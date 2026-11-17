"""Pytest configuration for the MrTotalVI test suite."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--seed", action="store", default=0, help="Seed set before each test.")


def pytest_configure(config):
    config.addinivalue_line("markers", "optional: mark test as optional.")
    config.addinivalue_line("markers", "slow: mark test as slow.")


@pytest.fixture(autouse=True)
def set_seed(request):
    """Set the scvi-tools seed before every test, as the scvi-tools suite does."""
    from scvi import settings

    settings.seed = int(request.config.getoption("--seed"))
