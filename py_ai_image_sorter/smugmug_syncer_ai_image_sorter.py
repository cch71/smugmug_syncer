import glob
import io
import json
import os
import queue
import uuid
from dataclasses import dataclass
from multiprocessing import Process, cpu_count, freeze_support, JoinableQueue

import cv2
import dlib
import numpy as np
import polars as pl
import zstandard
from _dlib_pybind11 import rectangle
from dotenv import load_dotenv

from log_config import log

load_dotenv()

from file_locations import (
    SMUGMUG_LOCAL_FACE_IMAGE_DIR,
    SMUGMUG_LOCAL_FACE_TAGS_FILE,
    SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR,
    SMUGMUG_LOCAL_CACHE_PATH,
    SMUGMUG_LOCAL_CACHE_DETECTION_FILE_PREFIX,
)


@dataclass
class FaceDetection:
    """
    Container for a single face detection
    """

    id: str | None
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

CHECKPOINT_SAVE_SIZE = 1000


def try_img_save(image, face_rect, face_id):
    """
    Saves off the image of just the face.  For this stage it is putting them all in one folder
    when it is done another util will sort them into people folder
    :param image: image data
    :param face_rect: rect of the face in the image
    :param face_id: the name to call it
    :return: None
    """
    if not SMUGMUG_LOCAL_FACE_IMAGE_DIR.exists():
        SMUGMUG_LOCAL_FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    x, y = face_rect["x"], face_rect["y"]
    w, h = face_rect["w"], face_rect["h"]

    face_img_location = image[y: y + h, x: x + w]
    face_img_out_path = SMUGMUG_LOCAL_FACE_IMAGE_DIR.joinpath(f"{face_id}.jpg")
    try:
        cv2.imwrite(str(face_img_out_path), face_img_location)
    except Exception as e:
        log.warning(f"Failed to save face image: {face_img_out_path} due to: {e}.")


def process_image_worker(image_q, detection_q):
    """
    Takes an image path and
        - reads it in from input q
        - detects faces and creates a FaceDetection record
        - save off face image
        - pushes it into an out q for further analysis
    :param image_q: Input q of image paths to operate on
    :param detection_q: Output queue where detections will go
    :return: None
    """
    while True:
        try:
            image_path = image_q.get(
                timeout=10
            )  # timeout in case no entries ever come for some reason
            image_q.task_done()
            if image_path is None:
                break
        except queue.Empty:
            break

        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces: [rectangle] = DETECTOR(gray, 1)
            img_id = os.path.basename(image_path)
            if len(faces):
                log.info(f"Found {len(faces)} in image: {img_id}")
                for idx, face in enumerate(faces):
                    shape = SP(gray, face)
                    face_embedding = FACEREC.compute_face_descriptor(image, shape)
                    face_rect = {
                        "x": face.left(),
                        "y": face.top(),
                        "w": face.width(),
                        "h": face.height(),
                    }
                    face_str = json.dumps(face_rect)
                    face_id = f"{img_id}_{idx}"
                    try_img_save(image, face_rect, face_id)

                    detection = FaceDetection(
                        id=f"{img_id}_{idx}",
                        image=img_id,
                        embedding=np.array(face_embedding),
                        window_in_image_json=face_str,
                    )
                    detection_q.put(detection)
            else:
                detection = FaceDetection(
                    id=None,
                    image=img_id,
                    embedding=np.array([]),
                    window_in_image_json=None,
                )
                detection_q.put(detection)
        except Exception as e:
            log.warning(
                f"Failed to process image: {image_path} due to: {e}.  Skipping..."
            )


def do_faces_likely_match(embedding1, embedding2):
    """
    Determines is 2 embeddings are closely related
    :param embedding1:
    :param embedding2:
    :return:
    """
    if len(embedding1) == 0:
        return False

    distance = np.linalg.norm(embedding1 - embedding2)
    return distance <= 0.6


def new_embeddings_save_dict():
    return {"id": [], "image_id": [], "embedding": [], "window_in_image_json": []}


def checkpoint_save_data(embeddings_to_save, sorted_faces, is_final):
    """
    Called periodically to save off data so that if an error occurs, it doesn't have to start over
    :param embeddings_to_save: FaceDetection list to save off
    :param sorted_faces: This is the state of the faces sorted into close matches i.e. "people"
    :param is_final: for the last save the sorted_faces list are removed of duplicates
    :return: None
    """
    if not SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.exists():
        SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # if sorted_faces is empty don't save
    if len(sorted_faces) > 0:

        if is_final:
            log.info("Finalizing labels file")
            # need to weed out duplicates in final images array
            for k, v in sorted_faces.items():
                sorted_faces[k] = {
                    "embeddings_face_id": v["embeddings_face_id"],
                    "images": list(set(v["images"])),
                    "faces": list(set(v["faces"])),
                }

        sorted_faces_str = json.dumps(sorted_faces, indent=4)
        with open(SMUGMUG_LOCAL_FACE_TAGS_FILE, "w") as f:
            f.write(sorted_faces_str)
            f.flush()

    # if we are final len could be 0
    if len(embeddings_to_save["id"]):
        df = pl.DataFrame(embeddings_to_save)
        cache_file = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath(
            f"{SMUGMUG_LOCAL_CACHE_DETECTION_FILE_PREFIX}-{uuid.uuid4()}.parquet"
        )
        df.write_parquet(cache_file, compression="zstd")
        log.debug("Checkpoint saving data")


def load_saved_detections():
    """
    Loads any previously saved values
    :return: the sorted_faces and embedded_lookup dicts populated from data or empty
    """
    sorted_faces = {}
    embedding_lookup = {}
    # TODO: This is as robust if things are missing. Assuming if directory exists everything inside it is correct
    if not SMUGMUG_LOCAL_FACE_TAGS_FILE.exists():
        return sorted_faces, embedding_lookup

    with open(SMUGMUG_LOCAL_FACE_TAGS_FILE, "r") as f:
        sorted_faces = json.load(f)

    df = (
        pl.scan_parquet(SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath("*.parquet"))
        .select(["id", "embedding"])
        .collect()
    )
    for person, v in sorted_faces.items():
        id_to_find = v["embeddings_face_id"]
        found_embedding = (
            df.filter(pl.col("id") == id_to_find).select(["embedding"]).item(0, 0)
        )
        embedding_lookup[person] = np.array(found_embedding)

    return sorted_faces, embedding_lookup


def process_detection(detection_q):
    """
    Reads a FaceDetection from the detection_q and compares it with known 'people'
    after x detection it does a checkpoint save for recovery in case of crash
    :param detection_q: Input q of FaceDetection
    :return: None
    """
    sorted_faces, embedding_lookup = load_saved_detections()
    to_save = new_embeddings_save_dict()

    while True:
        try:
            detection = detection_q.get(timeout=20)
            detection_q.task_done()

            # save detection for saving
            to_save["id"].append(detection.id)
            to_save["image_id"].append(detection.image)
            to_save["embedding"].append(detection.embedding.tolist())
            to_save["window_in_image_json"].append(detection.window_in_image_json)

            # sort images
            if detection.id is not None:
                is_found = False
                for k, embedding in embedding_lookup.items():
                    if do_faces_likely_match(detection.embedding, embedding):
                        is_found = True
                        sorted_faces[k]["images"].append(detection.image)
                        sorted_faces[k]["faces"].append(detection.id)
                if not is_found:
                    new_person = str(uuid.uuid4())
                    sorted_faces[new_person] = {
                        "embeddings_face_id": detection.id,
                        "images": [detection.image],
                        "faces": [detection.id],
                    }
                    embedding_lookup[new_person] = detection.embedding

            # save off in chunks
            if len(to_save["id"]) > CHECKPOINT_SAVE_SIZE:
                checkpoint_save_data(to_save, sorted_faces, is_final=False)
                # since already saved reset save cache
                to_save = new_embeddings_save_dict()

        except queue.Empty:
            break

    checkpoint_save_data(to_save, sorted_faces, is_final=True)


def get_unique_image_list():
    """
    :return: Unique list of image ids that are to have detection ran against
    """
    with open(
            SMUGMUG_LOCAL_CACHE_PATH.joinpath(".smugmug_db", "album_image_map.db"), "rb"
    ) as f:
        text_buffer = io.BytesIO()
        decompressor = zstandard.ZstdDecompressor()
        decompressor.copy_stream(f, text_buffer)
        data = json.loads(text_buffer.getvalue().decode("utf-8"))
        unique_images = set()
        for image_list in data.values():
            unique_images = unique_images.union(
                [
                    i["ImageKey"]
                    for i in image_list
                    if i["Format"] != "MP4" and not i["IsVideo"]
                ]
            )
        return unique_images


def get_already_processed_image_list():
    """
    :return: set of image ids that have already been processed
    """
    already_processed_images = set()
    parquet_files = SMUGMUG_LOCAL_CACHE_DETECTIONS_DIR.joinpath("*.parquet")
    if len(glob.glob(str(parquet_files))) > 0:
        df = pl.scan_parquet(parquet_files)
        images_processed = df.select("image_id").collect().to_series().to_list()
        already_processed_images = already_processed_images.union(images_processed)
    return already_processed_images


def process_images(image_q):
    """
    Reads image list in, builds paths from them, and adds them to the q if they haven't already been processed
    :param image_q: q of images to be processed
    :return: None
    """
    images = get_unique_image_list()
    already_processed_images = get_already_processed_image_list()

    for idx, img_key in enumerate(images):
        if img_key in already_processed_images:
            log.info(f"Image: {img_key} already processed so skipping")
            continue

        image_path = SMUGMUG_LOCAL_CACHE_PATH.joinpath("artifacts", img_key)
        image_q.put(image_path)
        # if idx > 50:
        #     return


def start_worker(fn, args):
    """
    Starts a worker proc
    :param fn: Function to run in proc
    :param args: args for worker
    :return: worker id
    """
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
    for i in range(NUMBER_OF_PROCESSES - 2):
        workers.append(start_worker(process_image_worker, (img_q, detections_q)))

    workers.append(start_worker(process_detection, (detections_q,)))

    for w in workers:
        w.join()
    log.info("Finished processing Detections")
