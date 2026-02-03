import multiprocessing as mp

import metaworld  # noqa: F401

mp.set_start_method("spawn", force=True)
