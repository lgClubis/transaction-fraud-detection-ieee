import subprocess
from pathlib import Path

COMPETITION = "ieee-fraud-detection"

def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.check_call(cmd)

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download competition data
    run([
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(raw_dir),
    ])

    # Unzip all zip files in raw_dir
    for z in raw_dir.glob("*.zip"):
        print(f"Unzipping {z.name}")
        run([
            "powershell",
            "-Command",
            f"Expand-Archive -Force '{z}' '{raw_dir}'"
        ])

    print("Done. Check data/raw for CSV files.")

if __name__ == "__main__":
    main()
