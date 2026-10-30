from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
load_dotenv(ROOT_DIR / "backend" / ".env")


def _configured_path(variable: str, default: Path) -> Path:
    path = Path(os.getenv(variable, default))
    return path if path.is_absolute() else BACKEND_DIR / path


DATASET_PATH = _configured_path(
    "CINESCOPE_DATASET", ROOT_DIR / "ASSETS" / "movie_metadata.csv"
)
ARTIFACT_DIR = _configured_path("CINESCOPE_ARTIFACT_DIR", BACKEND_DIR / "artifacts")
MODEL_PATH = ARTIFACT_DIR / "random_forest_pipeline.joblib"
MODEL_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
