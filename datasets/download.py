"""Pre-fetch every default dataset in one go (useful before an offline benchmark run).

    python datasets/download.py                 # everything except Speech Commands
    python datasets/download.py --include-heavy  # also Speech Commands (~2.3GB)
"""

from __future__ import annotations

import argparse

from liquid_playground.data import (
    load_ett,
    load_ozone,
    load_person_activity,
    load_room_occupancy,
    load_sequential_mnist,
    load_speech_commands,
)

LIGHT_LOADERS = {
    "sequential_mnist": load_sequential_mnist,
    "ozone": load_ozone,
    "room_occupancy": load_room_occupancy,
    "ett": load_ett,
    "person_activity": load_person_activity,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-heavy", action="store_true", help="also fetch Speech Commands")
    args = parser.parse_args()

    for name, loader in LIGHT_LOADERS.items():
        print(f"[download] {name} ...")
        loader()
        print(f"[download] {name} done")

    if args.include_heavy:
        print("[download] speech_commands ...")
        load_speech_commands()
        print("[download] speech_commands done")


if __name__ == "__main__":
    main()
