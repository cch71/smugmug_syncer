import json
import shutil
import threading

from dotenv import load_dotenv

from log_config import log

load_dotenv()
from file_locations import (
    SMUGMUG_LOCAL_FACE_PEOPLE_DIR,
    SMUGMUG_LOCAL_FACE_IMAGE_DIR,
    SMUGMUG_LOCAL_FACE_TAGS_FILE,
)


class Worker(threading.Thread):
    def __init__(self, person_id, faces):
        """
        Populates a 'person' folder with the face images generated by a previous process
        :param person_id: ID of person in the label file
        :param faces: face id list associated with this 'person'
        :return: None
        """
        super().__init__()
        self.person_id = person_id
        self.faces = faces

    def run(self):
        # mkdir person_folder
        person_dir = SMUGMUG_LOCAL_FACE_PEOPLE_DIR.joinpath(self.person_id)
        log.info(f"Making person folder: {person_dir}")
        person_dir.mkdir(parents=True, exist_ok=True)
        for face_id in self.faces:
            # copy to folder
            face_id = face_id[len("face_"):]
            img_file_name = f"{face_id}.jpg"
            face_file_in = SMUGMUG_LOCAL_FACE_IMAGE_DIR.joinpath(img_file_name)
            face_file_out = person_dir.joinpath(img_file_name)
            if face_file_in.exists() and not face_file_out.exists():
                try:
                    log.info(f"Copying: {face_file_in} to {face_file_out}")
                    shutil.copy(face_file_in, face_file_out)
                except Exception as e:
                    log.warning(
                        f"Failed to copy person file: {face_file_in} due to: {e}"
                    )


if __name__ == "__main__":
    with open(SMUGMUG_LOCAL_FACE_TAGS_FILE, "r") as f:
        sorted_faces = json.load(f)

    # Create worker threads
    workers = []
    for person, v in sorted_faces.items():
        worker = Worker(person, v["faces"])
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    log.info("Finished copying images")
