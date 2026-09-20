import subprocess
import sys


def test_predict_cli(demo_fasta):
    result = subprocess.run(
        [sys.executable, "-m", "aldiscore.main", "predict", str(demo_fasta), "--drop-gaps"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    score = float(result.stdout)
    assert 0.0 <= score <= 1.0
