"""Self-check for FrameFolderCapture: run with `python -m whalevision.frame_source_test`."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import AppConfig
from .export import write_automatic_only_csv
from .frame_source import FrameFolderCapture, open_capture, use_sequential_indexing


def demo():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for sec in (3, 7, 10):
            cv2.imwrite(str(folder / f"{sec}.png"), np.full((4, 4, 3), sec, np.uint8))
        (folder / "notes.txt").write_text("not a frame")
        # AppleDouble-style resource-fork junk (network/macOS volumes create
        # these next to every real file): matching extension, unreadable as
        # an image, and would even parse "10" as its second if not filtered.
        (folder / "._10.png").write_text("not actually a png")

        cap = open_capture(str(folder))
        assert isinstance(cap, FrameFolderCapture)
        assert cap.isOpened()
        assert cap.available_seconds == {3, 7, 10}
        assert cap.get(cv2.CAP_PROP_FPS) == 1.0

        # set + read lands exactly on the requested second, then advances.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 7)
        ok, frame = cap.read()
        assert ok and frame[0, 0, 0] == 7
        assert cap.get(cv2.CAP_PROP_POS_FRAMES) == 8

        # grab()-then-read() (the read_frame_at fast path) lands on the
        # same frame set() would have produced.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3)
        for _ in range(4):
            assert cap.grab()
        ok, frame = cap.read()
        assert ok and frame[0, 0, 0] == 7

        # a second with no matching file reads as "no frame".
        cap.set(cv2.CAP_PROP_POS_FRAMES, 4)
        ok, frame = cap.read()
        assert not ok and frame is None

        cap.release()

        empty = open_capture(str(folder / "does-not-exist"))
        assert not empty.isOpened()

        # sequential mode (pixel + no telemetry): every image counts,
        # arbitrary names included, indexed 0..N-1 in alphabetical order.
        cv2.imwrite(str(folder / "weird_name.jpg"), np.full((4, 4, 3), 99, np.uint8))
        seq_cap = FrameFolderCapture(str(folder), sequential=True)
        assert seq_cap.available_seconds == {0, 1, 2, 3}
        names = sorted(seq_cap.all_filenames().values())
        assert names == ["10.png", "3.png", "7.png", "weird_name.jpg"]
        assert "._10.png" not in names

        pixel_no_csv = AppConfig(measurement_unit="Pixels", csv_path="")
        assert use_sequential_indexing(pixel_no_csv)
        assert not use_sequential_indexing(AppConfig(measurement_unit="Meters", csv_path=""))
        assert not use_sequential_indexing(
            AppConfig(measurement_unit="Pixels", csv_path="telemetry.csv")
        )

        # write_automatic_only_csv attaches the original filename per frame_pos.
        candidate_pool = pd.DataFrame(
            {
                "frame_pos": [0, 3],
                "auto_length": [12.5, 8.0],
                "_candidate_id": [0, 1],
                "_auto_mask_crop": [None, None],
            }
        )
        cfg = AppConfig(
            measurement_unit="Pixels",
            csv_path="",
            video_path=str(folder),
            video_id="demo",
        )
        with tempfile.TemporaryDirectory() as out_tmp:
            out_dir = Path(out_tmp)
            csv_path = write_automatic_only_csv(candidate_pool, out_dir, cfg)
            written = pd.read_csv(csv_path)
            assert list(written["frame_name"]) == ["10.png", "weird_name.jpg"]
            assert "_candidate_id" not in written.columns
            assert "_auto_mask_crop" not in written.columns

    print("frame_source self-check: OK")


if __name__ == "__main__":
    demo()
