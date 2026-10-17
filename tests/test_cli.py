import subprocess
import sys


def _run_predict(*args):
    return subprocess.run(
        [sys.executable, "-m", "aldiscore.main", "predict", *args],
        capture_output=True,
        text=True,
    )


def test_predict_cli(demo_fasta):
    result = _run_predict(str(demo_fasta), "--in-type", "AA")
    assert result.returncode == 0, result.stderr
    score = float(result.stdout)
    assert 0.0 <= score <= 1.0


def test_predict_cli_accepts_lowercase_in_type(demo_fasta):
    upper = _run_predict(str(demo_fasta), "--in-type", "AA")
    lower = _run_predict(str(demo_fasta), "--in-type", "aa")
    assert lower.returncode == 0, lower.stderr
    assert lower.stdout == upper.stdout


def test_predict_cli_requires_in_type(demo_fasta):
    result = _run_predict(str(demo_fasta))
    assert result.returncode == 2
    assert "--in-type" in result.stderr
