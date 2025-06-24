import subprocess
import sys


def test_prediction_output():
    result = subprocess.run([sys.executable, 'predict.py'], capture_output=True, text=True)
    assert result.stdout.strip() != ''
