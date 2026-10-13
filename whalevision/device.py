"""Picks the accelerator and loads the YOLO/SAM models only when needed.

The module-level variables below are private: only the functions in this file
read or write them. That is the usual way to keep one shared copy of a heavy
model around, and it does not need a class.
"""

import gc

import torch
from ultralytics import SAM, YOLO

_device: "str | int" = "cpu"
_yolo_model_path = None
_sam_model_path = None
_yolo_model = None
_sam_model = None


def choose_device(compute_device: str) -> "str | int":
    if compute_device.startswith("CPU"):
        return "cpu"
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init(compute_device: str, yolo_model_path: str, sam_model_path: str) -> "str | int":
    """Pick the accelerator for this session and remember the model paths."""
    global _device, _yolo_model_path, _sam_model_path, _yolo_model, _sam_model
    _device = choose_device(compute_device)
    print(f"Selected accelerator: {_device}")
    _yolo_model_path = yolo_model_path
    _sam_model_path = sam_model_path
    _yolo_model = None
    _sam_model = None
    return _device


def get_device() -> "str | int":
    return _device


def get_yolo_model():
    """Load YOLO the first time a step actually needs to run it."""
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLO model...")
        _yolo_model = YOLO(_yolo_model_path)
    return _yolo_model


def get_sam_model():
    """Load SAM the first time it is needed, so screening starts faster and
    uses less memory."""
    global _sam_model
    if _sam_model is None:
        print("Loading SAM model...")
        _sam_model = SAM(_sam_model_path)
    return _sam_model


def peek_sam_model():
    """Return SAM if it is already loaded, but never load it."""
    return _sam_model


def unload_inference_models():
    """Drop both models and their cache, for when we are only browsing candidates."""
    global _yolo_model, _sam_model
    for model in (_yolo_model, _sam_model):
        if model is not None:
            predictor = getattr(model, "predictor", None)
            if predictor is not None:
                predictor.results = None
                predictor.batch = None
    _yolo_model = None
    _sam_model = None
    clear_accelerator_memory()


def clear_accelerator_memory():
    """Free leftover results and the accelerator cache, keeping the models loaded."""
    gc.collect()
    if _device == "mps":
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass
    elif _device != "cpu":
        torch.cuda.empty_cache()


def release_prediction(model, clear_cache=True):
    """Drop the references to the last result that Ultralytics keeps around."""
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        predictor.results = None
        predictor.batch = None
    if clear_cache:
        clear_accelerator_memory()


def is_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    indicators = (
        "out of memory",
        "not enough memory",
        "invalid buffer size",
        "failed to allocate",
        "allocation failed",
        "mps backend out of memory",
    )
    return any(indicator in message for indicator in indicators)


def predict_adaptive(model, frame, requested_imgsz, model_name, **predict_kwargs):
    """Run the model, retrying at lower resolutions if the GPU runs out of memory.

    If `predict_kwargs` contains `tracker`, this calls
    `model.track(..., persist=True)` instead of `model.predict(...)`. Tracking
    gives each animal an ID that survives from one frame to the next, which is
    how several whales in the same video are kept apart.
    """
    fallback_sizes = (1280, 1024, 960, 800, 640, 512, 384)
    sizes = [requested_imgsz]
    sizes.extend(size for size in fallback_sizes if size < requested_imgsz)
    last_error_message = ""
    tracker = predict_kwargs.pop("tracker", None)

    for attempt, image_size in enumerate(dict.fromkeys(sizes), start=1):
        if attempt > 1:
            clear_accelerator_memory()
        try:
            predict_fn = (
                (lambda *fn_args, **kwargs: model.track(*fn_args, persist=True, tracker=tracker, **kwargs))
                if tracker is not None
                else model.predict
            )
            result = predict_fn(
                frame,
                device=_device,
                verbose=False,
                imgsz=image_size,
                **predict_kwargs,
            )[0]
            if attempt > 1:
                print(
                    f"  > {model_name}: inference succeeded at {image_size}px "
                    "after automatic reduction for memory."
                )
            return result
        except RuntimeError as error:
            if not is_memory_error(error):
                raise
            last_error_message = str(error)
            print(
                f"  > Insufficient memory with {model_name} at {image_size}px; "
                "freeing the cache and trying a lower resolution."
            )
            if getattr(model, "predictor", None) is not None:
                model.predictor = None

    raise RuntimeError(
        f"Insufficient memory even at the minimum resolution for {model_name}. "
        "Restart choosing 'CPU (lower memory risk)' or reduce the quality. "
        f"Last error: {last_error_message}"
    )
