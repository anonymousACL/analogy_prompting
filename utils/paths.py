from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

ROOT = get_project_root()
IMG_DIR = ROOT / 'data/images/arrows'