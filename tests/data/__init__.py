"""Builds a path to the test data directory."""

from pathlib import Path


TEST_DATA_PATH = Path(__file__).resolve().parent
