import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs-cal"))
