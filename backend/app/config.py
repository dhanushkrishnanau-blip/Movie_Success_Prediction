from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(
    os.getenv("CINESCOPE_DATASET", ROOT_DIR / "ASSETS" / "movie_metadata.csv")
)
ARTIFACT_DIR = Path(
    os.getenv("CINESCOPE_ARTIFACT_DIR", ROOT_DIR / "backend" / "artifacts")
)
MODEL_PATH = ARTIFACT_DIR / "random_forest_pipeline.joblib"
MODEL_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
