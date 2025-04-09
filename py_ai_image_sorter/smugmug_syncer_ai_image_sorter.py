import dlib
import cv2
import os
import numpy as np
from _dlib_pybind11 import rectangle
import zstandard
import json
import io
import queue
from dataclasses import dataclass
from multiprocessing import Process, cpu_count, freeze_support, JoinableQueue
import logging
import time
from dotenv import load_dotenv
from pathlib import Path
import polars as pl
import uuid

load_dotenv()

class UtcFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = time.gmtime(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%dT%H:%M:%S", ct)
            s = f"{t}.{int(record.msecs):03d}Z"
        return s


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = UtcFormatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
log.addHandler(handler)

@dataclass
class FaceDetection:
    image: str
    embedding: np.ndarray
    window_in_image_json: str | None

# https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
# Load dlib models
DETECTOR = dlib.get_frontal_face_detector()
SP = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
FACEREC = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Num of CPUs on system
NUMBER_OF_PROCESSES = cpu_count()

SMUGMUG_LOCAL_CACHE_PATH = Path(os.getenv("SMUGMUG_SYNC_LOCATION"))
SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR = SMUGMUG_LOCAL_CACHE_PATH.joinpath("face_detections")
SMUGMUG_LOCAL_CACHE_DETECTION_FILE_PREFIX = "detections"
SMUGMUG_LOCAL_FACE_TAGS_FILE = SMUGMUG_LOCAL_CACHE_PATH.joinpath("face_detections", "labels.json")

CHECKPOINT_SAVE_SIZE = 500

def process_image_worker(image_q, detection_q):
    while True:
        try:
            image_path = image_q.get(timeout = 10) # timeout in case no entries ever come for some reason
            image_q.task_done()
            if image_path is None:
                break
        except queue.Empty:
            break

        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces:[rectangle] = DETECTOR(gray, 1)
            img_id = os.path.basename(image_path)
            if len(faces):
                log.info(f"Found {len(faces)} in image: {img_id}")
                for face in faces:
                    shape = SP(gray, face)
                    face_embedding = FACEREC.compute_face_descriptor(image, shape)
                    face_str = json.dumps({
                        "t": face.top(),
                        "b": face.bottom(),
                        "l": face.left(),
                        "r": face.right(),
                    })
                    detection = FaceDetection(image=img_id, embedding=np.array(face_embedding), window_in_image_json=face_str)
                    detection_q.put(detection)
            else:
                detection = FaceDetection(image=img_id, embedding=np.array([]), window_in_image_json=None)
                detection_q.put(detection)
        except Exception as e:
            log.warning(f"Failed to process image: {image_path} due to: {e}.  Skipping...")

def do_faces_likely_match(embedding1, embedding2):
    if len(embedding1) == 0:
        return False

    distance = np.linalg.norm(embedding1 - embedding2)
    return distance <= 0.6

def new_embeddings_save_dict():
    return {
        "image_id": [],
        "embedding": [],
        "window_in_image_json": []
    }

def checkpoint_save_data(embeddings_to_save, sorted_faces, is_final):
    if not SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.exists():
        SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if is_final:
        # need to week out duplicates in final images array
        for k,images in sorted_faces:
            sorted_faces[k] = list(set(images))

    sorted_faces_str = json.dumps(sorted_faces, indent=4)
    with open(SMUGMUG_LOCAL_FACE_TAGS_FILE, "w") as f:
        f.write(sorted_faces_str)
        f.flush()

    # if we are final len could be 0
    if len(embeddings_to_save["image_id"]):
        df = pl.DataFrame(embeddings_to_save)
        cache_file = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath(f"{SMUGMUG_LOCAL_CACHE_DETECTION_FILE_PREFIX}-{uuid.uuid4()}.parquet")
        df.write_parquet(cache_file, compression="zstd")
        log.debug("Checkpoint saving data")

def load_saved_detections():
    sorted_faces = {}
    embedding_lookup = {}
    # TODO: This is as robust if things are missing. Assuming if directory exists everything inside it is correct
    if not SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.exists():
        return sorted_faces, embedding_lookup

    with open(SMUGMUG_LOCAL_FACE_TAGS_FILE, 'r') as f:
        sorted_faces = json.load(f)

    df = pl.scan_parquet(SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath("*.parquet")).select(["image_id", "embedding"]).collect()
    for person, images in sorted_faces.items():
        image_to_find = images[0]
        found_embedding = df.filter(pl.col("image_id") == image_to_find).select(["embedding"]).item(0,0)
        embedding_lookup[person] = np.array(found_embedding)

    return sorted_faces, embedding_lookup


def process_detection(detection_q):
    sorted_faces, embedding_lookup = load_saved_detections()
    to_save = new_embeddings_save_dict()


    while True:
        try:
            detection = detection_q.get(timeout = 20)
            detection_q.task_done()

            # save detection for saving
            to_save["image_id"].append(detection.image)
            to_save["embedding"].append(detection.embedding.tolist())
            to_save["window_in_image_json"].append(detection.window_in_image_json)

            # sort images
            if len(detection.embedding):
                is_found = False
                for k, embedding in embedding_lookup.items():
                    if do_faces_likely_match(detection.embedding, embedding):
                        is_found = True
                        sorted_faces[k].append(detection.image)
                if not is_found:
                    new_person = str(uuid.uuid4())
                    sorted_faces.setdefault(new_person, []).append(detection.image)
                    embedding_lookup[new_person] = detection.embedding

            # save off in chunks
            if len(to_save["image_id"]) > CHECKPOINT_SAVE_SIZE:
                checkpoint_save_data(to_save, sorted_faces, is_final = False)
                # since already saved reset save cache
                to_save = new_embeddings_save_dict()

        except queue.Empty:
            break

    checkpoint_save_data(to_save, sorted_faces, is_final=True)


def get_unique_image_list():
    with open(SMUGMUG_LOCAL_CACHE_PATH.joinpath(".smugmug_db","album_image_map.db"), 'rb') as f:
        text_buffer = io.BytesIO()
        decompressor = zstandard.ZstdDecompressor()
        decompressor.copy_stream(f, text_buffer)
        data = json.loads(text_buffer.getvalue().decode('utf-8'))
        new_data = {}
        for image_list in data.values():
            new_data.update({str(i["ImageKey"]): i for i in image_list})
        return new_data.values()

def process_images(image_q):
    images = get_unique_image_list()
    already_processed_images = set()

    if SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.exists():
        df = pl.scan_parquet(SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath("*.parquet"))
        images_processed = df.select("image_id").collect().to_series().to_list()
        already_processed_images = already_processed_images.union(images_processed)

    for idx, img in enumerate(images):
        if img["Format"] != "MP4" and not img["IsVideo"]:
            # print(json.dumps(img, indent=4))
            img_key = img['ImageKey']
            if img_key in already_processed_images:
                log.info(f"Image: {img_key} already processed so skipping")
                continue

            image_path = SMUGMUG_LOCAL_CACHE_PATH.joinpath("artifacts", img_key)
            image_q.put(image_path)
            # if idx > 50:
            #     return

def start_worker(fn, args):
    p = Process(target=fn, args=args)
    p.start()
    return p

if __name__ == "__main__":
    freeze_support()

    img_q = JoinableQueue()
    detections_q = JoinableQueue()

    workers = []
    workers.append(start_worker(process_images, (img_q,)))

    workers = []
    for i in range(NUMBER_OF_PROCESSES-2):
        workers.append(start_worker(process_image_worker, (img_q,detections_q)))
        #Process(target=, args=(img_q, detections_q)).start()

    workers.append(start_worker(process_detection, (detections_q,)))
    # Process(target=process_detection, args=(detections_q,)).start()

    for w in workers:
        w.join()
    log.info("Finished processing Detections")


