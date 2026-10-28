"""
Lock ridge_gbm/protocol.json by recording its SHA-256 (run.py refuses to run if it changes).

Usage (from backend/):  python -m ml.v2.ridge_gbm.lock [--force]
"""

import argparse

from ml.v2.common import lock_protocol
from ml.v2.ridge_gbm.config import PROTOCOL_FILE, PROTOCOL_LOCK

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lock ml/v2/ridge_gbm/protocol.json")
    parser.add_argument("--force", action="store_true", help="deliberately re-lock (the new lock time is visible)")
    lock = lock_protocol(force=parser.parse_args().force, protocol_file=PROTOCOL_FILE, lock_file=PROTOCOL_LOCK)
    print(f"Protocol locked: sha256 {lock['sha256']} at {lock['locked_at']}")
