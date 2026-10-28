"""
Lock the pre-registered protocol (ml/v2/protocol.json) by recording its SHA-256.

Every v2 experiment and the supplementary evaluation refuse to run if the
protocol changed after it was locked.

Usage (from the backend folder):  python -m ml.v2.lock_protocol [--force]
"""

import argparse

from ml.v2.common import lock_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lock ml/v2/protocol.json")
    parser.add_argument("--force", action="store_true", help="deliberately re-lock (the new lock time is visible)")
    lock = lock_protocol(force=parser.parse_args().force)
    print(f"Protocol locked: sha256 {lock['sha256']} at {lock['locked_at']}")
