"""Application settings, and saving/reloading the last session's settings."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent.parent
LAST_CONFIG_PATH = SCRIPT_DIR / ".whale_vision_last_config.json"


@dataclass
class AppConfig:
    measurement_unit: str = "Meters"
    input_mode: str = "Video"
    video_path: str = ""
    csv_path: str = ""
    video_time_column: str = "sec_video"
    altitude_column: str = "laser_altitude_fixed"
    flight_date_column: str = ""
    output_folder: str = ""
    video_id: str = ""
    yolo_model_path: str = str(SCRIPT_DIR / "best-seg.pt")
    sam_model_path: str = str(SCRIPT_DIR / "sam2_b.pt")
    start_sec: float = 15.0
    end_sec: float = 180.0
    sensor_width_mm: float = 17.29
    image_width_px: int = 3840
    focal_length_mm: float = 12.1086484375
    confidence: float = 0.8
    analysis_imgsz: int = 640
    review_imgsz: int = 640
    sam_box_margin: int = 10
    compute_device: str = "Automatic"
    pipeline_mode: str = "YOLO + SAM"
    scan_fps: float = 2.0
    max_alt_diff: float = 0.5
    use_tilt_filter: bool = False
    max_tilt_deg: float = 0.0
    central_fraction: float = 0.5
    use_mean_mask_filter: bool = True
    mean_mask_min_iou: float = 0.55
    mean_mask_warmup: int = 10
    use_candidate_preview: bool = True
    skip_review: bool = False
    expected_whale_count: int = 1
    max_accepted: int = 30
    max_frames_per_sec: int = 1
    target_quartiles: str = "3,4"
    yolo_dataset_folder: str = ""
    custom_segment_names: str = ""


def load_last_configuration() -> AppConfig:
    if not LAST_CONFIG_PATH.is_file():
        return AppConfig()
    try:
        saved = json.loads(LAST_CONFIG_PATH.read_text(encoding="utf-8"))
        valid_fields = AppConfig.__dataclass_fields__
        return AppConfig(**{key: value for key, value in saved.items() if key in valid_fields})
    except (OSError, ValueError, TypeError):
        return AppConfig()


def save_last_configuration(config: AppConfig) -> None:
    LAST_CONFIG_PATH.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
