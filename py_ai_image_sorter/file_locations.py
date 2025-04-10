import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SMUGMUG_LOCAL_CACHE_PATH = Path(os.getenv("SMUGMUG_SYNC_LOCATION"))
SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR = SMUGMUG_LOCAL_CACHE_PATH.joinpath(
    "face_detections"
)

# This is the filename prefix that detections will be stored into
SMUGMUG_LOCAL_CACHE_DETECTION_FILE_PREFIX = "detections"
SMUGMUG_LOCAL_FACE_TAGS_FILE = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath(
    "labels.json"
)
SMUGMUG_LOCAL_FACE_IMAGE_DIR = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath(
    "face_images"
)

SMUGMUG_LOCAL_FACE_PEOPLE_DIR = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath("people")
