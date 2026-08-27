import os
import sys

# Ensure this package directory is on sys.path so the library's internal
# flat imports (e.g. `import utils` inside model.py / visualize.py) resolve
# to the modules that live alongside this file.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
