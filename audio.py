#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

TRACKS_DIR = Path("tracks")

def check_ffmpeg():
    """Check whether ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def convert_mp3_to_wav(input_mp3: Path, output_wav: Path | None = None) -> bool:
    """Convert an MP3 file to WAV using ffmpeg."""
    if output_wav is None:
        output_wav = input_mp3.with_suffix(".wav")

    print(f"Converting: {input_mp3.name} → {output_wav.name}")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # overwrite automatically
                "-i", str(input_mp3),
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                str(output_wav)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode == 0:
            print(f"✔ Success: Created {output_wav.name}")
            return True
        else:
            print("✘ ffmpeg error:\n", result.stderr.decode())
            return False

    except Exception as e:
        print("✘ Exception:", e)
        return False


def convert_tracks_folder():
    """Convert all MP3 files in the tracks folder to WAV."""
    TRACKS_DIR.mkdir(exist_ok=True)

    mp3_files = list(TRACKS_DIR.glob("*.mp3"))

    if not mp3_files:
        print("⚠ No MP3 files found in /tracks")
        return

    print(f"Found {len(mp3_files)} MP3 file(s). Starting conversion...\n")

    for file in mp3_files:
        convert_mp3_to_wav(file)


def convert_single_file():
    """Prompt for a file path and convert it."""
    path = input("\nEnter the path to the MP3 file: ").strip()
    file = Path(path)

    if not file.exists():
        print("✘ Error: File not found.")
        return

    if file.suffix.lower() != ".mp3":
        print("✘ Error: File must be an .mp3")
        return

    convert_mp3_to_wav(file)


def show_menu():
    print("=" * 55)
    print("        Hand DJ — MP3 → WAV Conversion Tool")
    print("=" * 55)
    print("This tool prepares audio files for use in the Hand DJ app.\n")
    print("1. Convert ALL MP3s in /tracks")
    print("2. Convert ONE MP3 file")
    print("3. Exit\n")

    return input("Choose an option (1–3): ").strip()


def main():
    if not check_ffmpeg():
        print("✘ ffmpeg is not installed.")
        print("Install it using one of the following:\n"
              "  macOS:  brew install ffmpeg\n"
              "  Debian/Ubuntu: sudo apt install ffmpeg\n"
              "  Windows: Download from ffmpeg.org\n")
        return

    while True:
        choice = show_menu()

        if choice == "1":
            convert_tracks_folder()
        elif choice == "2":
            convert_single_file()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please pick 1–3.\n")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
