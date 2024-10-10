import atexit
import json
import logging
import os
import sys

import numpy as np
import open3d as o3d
from termcolor import colored

__all__ = [
    "setup_logger",
]


def save_detection_list_to_json(detection_list, output_file):
    def serialize_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def serialize_open3d(obj):
        if isinstance(obj, o3d.geometry.AxisAlignedBoundingBox):
            return {
                "type": "AxisAlignedBoundingBox",
                "min_bound": obj.min_bound.tolist(),
                "max_bound": obj.max_bound.tolist(),
            }
        elif isinstance(obj, o3d.geometry.OrientedBoundingBox):
            return {
                "type": "OrientedBoundingBox",
                "center": obj.center.tolist(),
                "extent": obj.extent.tolist(),
                "R": obj.R.tolist(),
            }
        elif isinstance(obj, o3d.geometry.PointCloud):
            return {
                "type": "PointCloud",
                "points": np.asarray(obj.points).tolist(),
                "colors": np.asarray(obj.colors).tolist() if obj.has_colors() else None,
                "normals": np.asarray(obj.normals).tolist() if obj.has_normals() else None,
            }
        return obj

    def serialize_detection(detection):
        serialized = {}
        for key, value in detection.items():
            if key in ["axis_aligned_bbox", "oriented_bbox", "pcd"]:
                serialized[key] = serialize_open3d(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = serialize_numpy(value)
            elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)  # Convert other types to string
        return serialized

    serialized_list = [serialize_detection(detection) for detection in detection_list]

    with open(output_file, "w") as f:
        json.dump(serialized_list, f, indent=2)

    print(f"Detection list saved to {output_file}")


class SkipImageException(Exception):
    def __init__(self, message="Known exception, skip the image."):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(output=None, distributed_rank=0, *, name="metricdepth", color=True, abbrev_name=None):
    """Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        abbrev_name (str): an abbreviation of the module, to avoid log names in logs.
            Set to "" not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # NOTE: if more detailed, change it to logging.DEBUG
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "d2"

    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s %(message)s ", datefmt="%m/%d %H:%M:%S")
    # stdout logging: master  only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)  # NOTE: if more detailed, change it to logging.DEBUG
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.INFO)  # NOTE: if more detailed, change it to logging.DEBUG
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


from iopath.common.file_io import PathManager as PathManagerBase

PathManager = PathManagerBase()


# cache the opened file object, so that different calls to 'setup_logger
# with the same file name can safely write to the same file.
def _cached_log_stream(filename):
    # use 1K buffer if writting to cloud storage
    io = PathManager.open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io
