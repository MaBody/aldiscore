from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def demo_fasta() -> Path:
    return REPO_ROOT / "demo" / "data" / "protein.0.fasta"


@pytest.fixture(scope="session")
def dna_seqs() -> list[str]:
    return [
        "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTAGCTAGGCTAGCTAGCTAGCATCG",
        "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTAGCTAGGCTAGCTAGCTAGCTTCG",
        "ATGCGTACGTTAGCATCGATCGATCGTAGCTAGCTTGCTAGGCTAGCTAGCTAGCATCG",
        "ATGCGTACGTTAGCATCGATCGATCGTAGCAAGCTAGCTAGGCTAGCTAGCTAGCATCG",
    ]
