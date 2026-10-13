"""Pytest bootstrap for litellm tests.

Makes the `litellm/` directory importable so tests can `import
update_free_models` without packaging the repo.
"""

import sys
from pathlib import Path

LITELLM_DIR = Path(__file__).resolve().parent.parent
if str(LITELLM_DIR) not in sys.path:
    sys.path.insert(0, str(LITELLM_DIR))
