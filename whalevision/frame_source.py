"""Opens the source of frames: a real video, or a folder of images.

A folder is expected to hold one image per second of video, with the second
written in the file name.

`FrameFolderCapture` copies the small part of the `cv2.VideoCapture` API that
screening.py and review.py use (isOpened, get/set CAP_PROP_FPS and
CAP_PROP_POS_FRAMES, grab, read, release), so the rest of the pipeline never
has to know which kind of source it is reading. Its fps is 1.0, which makes
`frame_pos` the same number as the second, so nothing else needs to convert.
"""

import re
from pathlib import Path

import cv2

_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg")
_TRAILING_DIGITS = re.compile(r"(\d+)$")


class FrameFolderCapture:
    def __init__(self, folder_path, sequential=False):
        """With `sequential=True` the images are numbered 0, 1, 2, ... in
        alphabetical order instead of by the second read from the file name.
        That is used in pixel mode without telemetry (see
        `use_sequential_indexing`), where file names follow no pattern and
        every image in the folder has to be analyzed anyway."""
        # Skip files starting with "." whatever their extension. macOS and
        # network drives often leave a "._somename.png" next to every real
        # file: same extension, but not an image. cv2.imread fails on those,
        # and in sequential mode (where every file counts) that failure used
        # to reach screening.
        image_paths = sorted(
            path
            for path in Path(folder_path).iterdir()
            if path.suffix.lower() in _FRAME_EXTENSIONS and not path.name.startswith(".")
        )
        if sequential:
            self._frames = dict(enumerate(image_paths))
        else:
            self._frames = {}
            skipped = []
            for path in image_paths:
                match = _TRAILING_DIGITS.search(path.stem)
                if not match:
                    skipped.append(path.name)
                    continue
                self._frames[int(match.group(1))] = path
            if skipped:
                print(
                    f"  > Frame folder: ignored {len(skipped)} file(s) without a "
                    f"recognizable second in the name: {', '.join(skipped[:5])}"
                    + ("…" if len(skipped) > 5 else "")
                )
        self._pos = 0

    @property
    def available_seconds(self):
        return set(self._frames)

    def all_filenames(self):
        """Map frame_pos -> file name, so results can be traced back to the
        image they came from."""
        return {position: path.name for position, path in self._frames.items()}

    def isOpened(self):
        return bool(self._frames)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 1.0
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
        return True

    def grab(self):
        self._pos += 1
        return True

    def read(self):
        # Same rule as cv2.VideoCapture: CAP_PROP_POS_FRAMES is the index of
        # the NEXT frame to read, so read() takes that one and then steps
        # forward. Keep this in step with grab() and set(), which read_frame_at
        # uses together.
        path = self._frames.get(self._pos)
        self._pos += 1
        if path is None:
            return False, None
        image = cv2.imread(str(path))
        if image is None:
            return False, None
        return True, image

    def release(self):
        pass


def open_capture(source_path, sequential=False):
    """Open `source_path` as a video, or as a folder of frames if it is a
    directory (see FrameFolderCapture)."""
    if Path(source_path).is_dir():
        return FrameFolderCapture(source_path, sequential=sequential)
    return cv2.VideoCapture(source_path)


def use_sequential_indexing(config):
    """True in pixel mode without telemetry.

    There is no second or altitude to pick frames by, so every image in the
    folder is analyzed whatever it is called, numbered in sequence rather than
    by a second parsed from the file name.
    """
    return config.measurement_unit == "Pixels" and not config.csv_path
