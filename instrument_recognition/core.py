"""core.py - root module exports"""


from pathlib import Path


###############################################################################
# Constants
###############################################################################


# Static directories
ASSETS_DIR = Path(__file__).parent / 'assets'
CACHE_DIR = Path(__file__).parent.parent / 'cache'
DATA_DIR = Path(__file__).parent.parent / 'data'
LOG_DIR = Path(__file__).parent / 'test-tubes'


# Static, module-wide constants
RANDOM_SEED = 20
SAMPLE_RATE = 48000