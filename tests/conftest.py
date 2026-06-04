import sys
from pathlib import Path

# Make the Skylance project root importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
