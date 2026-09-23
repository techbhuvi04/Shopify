"""Multimodal Product Matching & Retrieval System."""

import os
import sys

if sys.platform == "darwin":
    # The macOS wheels of torch and faiss-cpu each bundle their own libomp.
    # Loading both in one process aborts ("OMP: Error #15") or segfaults during
    # multithreaded FAISS search. The stable combination is: allow the
    # duplicate runtime, load torch's libomp first, and run FAISS with a single
    # OpenMP thread (exact flat search is fast at this scale anyway).
    # Override the FAISS thread count with MPM_FAISS_THREADS if needed.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch  # noqa: F401  (import order matters)
    import faiss

    faiss.omp_set_num_threads(int(os.environ.get("MPM_FAISS_THREADS", "1")))

__version__ = "0.1.0"
