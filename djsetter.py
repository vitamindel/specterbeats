#!/usr/bin/env python3
"""
Specter Beats Launcher
Initializes the motion-controlled DJ system and handles track discovery.
"""

import argparse
import os
import random
import sys


# ---------------------------------------------------------
# Track Discovery Utilities
# ---------------------------------------------------------

def list_stem_folders(base_path="tracks"):
    """Return folders inside /tracks that include both vocal + instrumental stems."""
    if not os.path.isdir(base_path):
        print(f"❌ The directory '{base_path}' does not exist.")
        return []

    stem_folders = []
    for entry in sorted(os.listdir(base_path)):
        full = os.path.join(base_path, entry)
        if not os.path.isdir(full):
            continue

        if stems_present(full):
            stem_folders.append(entry)

    return stem_folders


def stems_present(folder):
    """
    Determine whether a folder contains 'vocals*' and 'instrumental*'
    in accepted audio formats.
    """
    try:
        contents = os.listdir(folder)
    except Exception:
        return False

    audio_extensions = (".mp3", ".wav", ".flac", ".m4a", ".aac")

    has_v = any(
        f.lower().startswith("vocals") and f.lower().endswith(audio_extensions)
        for f in contents
    )
    has_i = any(
        f.lower().startswith("instrumental") and f.lower().endswith(audio_extensions)
        for f in contents
    )

    return has_v and has_i


# ---------------------------------------------------------
# Interactive Prompt Flow
# ---------------------------------------------------------

def choose_tracks_interactively():
    """Provide a selection list and return two chosen tracks."""
    options = list_stem_folders()

    if not options:
        print("❌ No compatible tracks detected inside /tracks.")
        return None, None

    print(f"\n🎵 Available Track Sets ({len(options)})")
    print("-" * 48)
    for idx, name in enumerate(options, 1):
        print(f" {idx:2d}. {name}")
    print()

    def pick(label):
        while True:
            try:
                raw = input(f"Select for {label} (1-{len(options)} or Enter for random): ").strip()
                if raw == "":
                    pick = random.choice(options)
                    print(f"🎲 Randomized: {pick}")
                    return pick

                number = int(raw) - 1
                if 0 <= number < len(options):
                    return options[number]

                print(f"❌ Choose a number between 1 and {len(options)}.")

            except ValueError:
                print("❌ Invalid input — enter a number.")
            except (KeyboardInterrupt, EOFError):
                print("\n❌ Selection cancelled.")
                return None

    left = pick("DECK A")
    if left is None:
        return None, None

    right = pick("DECK B")
    if right is None:
        return None, None

    return left, right


# ---------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Specter Beats – Motion DJ Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python specter_beats.py               -> Standard mode
  python specter_beats.py --nosync      -> Disable BPM sync
  python specter_beats.py --auto        -> Skip selection (auto-load)
        """
    )

    parser.add_argument("--nosync", action="store_true", help="Disable BPM alignment between decks")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompts and load defaults")

    args = parser.parse_args()

    bpm_sync_enabled = not args.nosync
    interactive_mode = not args.auto

    print("\n🕯️  Specter Beats — Gesture-Powered DJ Engine")
    print("✋ Controls via hand tracking • Press 'q' to exit\n")

    if not bpm_sync_enabled:
        print("🔄 BPM SYNC: OFF\n")

    try:
        # Import runtime controller
        from dj_controller import DeckMaster

        # Select tracks
        selections = None
        if interactive_mode:
            selections = choose_tracks_interactively()

        print("🎧 Initializing audio engine...")
        engine = DeckMaster(enable_bpm_sync=bpm_sync_enabled, selected_tracks=selections)
        engine.run()

    except ImportError as err:
        print(f"\n❌ Dependency missing: {err}")
        print("Activate the environment:")
        print("  source venv_py311/bin/activate")
        print("  python specter_beats.py")
        sys.exit(1)

    except Exception as exc:
        print(f"\n❌ Error starting Specter Beats: {exc}")
        sys.exit(1)


# ---------------------------------------------------------

if __name__ == "__main__":
    main()
