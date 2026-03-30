from __future__ import annotations

import ast
import hashlib
import math
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
import types
import wave
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    import comfy
    import comfy.model_management as mm
    import comfy.sd
    import comfy.utils
except Exception:  # pragma: no cover
    comfy = None
    mm = None

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

try:
    from comfy_execution.graph import ExecutionBlocker
    from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:  # pragma: no cover
    ExecutionBlocker = None
    GraphBuilder = None

    def is_link(value):
        return isinstance(value, (list, tuple)) and len(value) >= 2


MAX_FLOW_NUM = 10
MAX_RESOLUTION = 16384
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


def _require(name: str, value: Any) -> Any:
    if value is None:
        raise RuntimeError(f"{name} is required for this node.")
    return value


def _safe_eval(expression: str, values: dict[str, Any]) -> Any:
    allowed_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "int": int,
        "float": float,
        "bool": bool,
    }
    allowed_names = {"pi": math.pi, "euler": math.e, "True": True, "False": False, **values}
    allowed_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a**b,
        ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b,
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Name '{node.id}' is not allowed")
            return allowed_names[node.id]
        if isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator '{type(node.op).__name__}' is not allowed")
            return op(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator '{type(node.op).__name__}' is not allowed")
            return op(evaluate(node.operand))
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                op = allowed_ops.get(type(op_node))
                if op is None or not op(left, evaluate(comparator)):
                    return False
                left = evaluate(comparator)
            return True
        if isinstance(node, ast.BoolOp):
            values_eval = [bool(evaluate(v)) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values_eval)
            if isinstance(node.op, ast.Or):
                return any(values_eval)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            fn = allowed_functions.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function '{node.func.id}' is not allowed")
            return fn(*[evaluate(arg) for arg in node.args])
        raise ValueError(f"Expression node '{type(node).__name__}' is not supported")

    tree = ast.parse(expression, mode="eval")
    return evaluate(tree.body)


def _input_directory() -> str:
    if folder_paths is not None:
        return folder_paths.get_input_directory()
    return os.path.join(os.getcwd(), "input")


def _repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def _sample_input_directory() -> pathlib.Path:
    return _repository_root() / "samples" / "input"


def _repo_relative_path(path: pathlib.Path) -> str:
    return path.relative_to(_repository_root()).as_posix()


def _output_directory(save_output: bool) -> str:
    if folder_paths is not None:
        return folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    return tempfile.gettempdir()


def _resolve_input_path(name_or_path: str) -> str:
    if not name_or_path:
        raise ValueError("A file or directory name is required.")
    if os.path.isabs(name_or_path):
        return name_or_path
    repo_candidate = _repository_root() / name_or_path
    if repo_candidate.exists():
        return str(repo_candidate)
    sample_candidate = _sample_input_directory() / name_or_path
    if sample_candidate.exists():
        return str(sample_candidate)
    if folder_paths is not None:
        try:
            return folder_paths.get_annotated_filepath(name_or_path)
        except Exception:
            pass
    return os.path.join(_input_directory(), name_or_path)


def _list_input_audio_files() -> list[str]:
    files = set()

    input_dir = pathlib.Path(_input_directory())
    if input_dir.is_dir():
        for item in input_dir.iterdir():
            if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
                files.add(item.name)

    sample_input_dir = _sample_input_directory()
    if sample_input_dir.is_dir():
        for item in sample_input_dir.iterdir():
            if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
                files.add(_repo_relative_path(item))

    return sorted(files, key=str.casefold)


def _list_input_subdirectories() -> list[str]:
    directories = set()

    input_dir = pathlib.Path(_input_directory())
    if input_dir.is_dir():
        for item in input_dir.iterdir():
            if item.is_dir() and item.name != "clipspace":
                directories.add(item.name)

    sample_input_dir = _sample_input_directory()
    if sample_input_dir.is_dir():
        for item in sample_input_dir.iterdir():
            if item.is_dir() and item.name != "clipspace":
                directories.add(_repo_relative_path(item))

    return sorted(directories, key=str.casefold)


def _hash_path(path: str) -> str:
    target = pathlib.Path(path)
    if not target.exists():
        return "missing"
    if target.is_file():
        stat = target.stat()
        return f"{target}:{stat.st_mtime_ns}:{stat.st_size}"
    digest = hashlib.sha256()
    for child in sorted(target.glob("**/*")):
        if child.is_file():
            stat = child.stat()
            digest.update(f"{child.relative_to(target)}:{stat.st_mtime_ns}:{stat.st_size}\n".encode("utf8"))
    return digest.hexdigest()


def _load_wav_fallback(path: str, frame_offset: int = 0, num_frames: int = 0):
    _require("torch", torch)
    with wave.open(path, "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        sample_rate = int(wav_file.getframerate())
        total_frames = int(wav_file.getnframes())
        wav_file.setpos(min(max(int(frame_offset), 0), total_frames))
        frames_to_read = max(0, total_frames - wav_file.tell()) if int(num_frames) <= 0 else int(num_frames)
        raw = wav_file.readframes(frames_to_read)

    if np is not None:
        if sample_width == 1:
            waveform = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(torch.float32)
            waveform = (waveform - 128.0) / 128.0
        elif sample_width == 2:
            waveform = torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).to(torch.float32) / 32768.0
        elif sample_width == 4:
            waveform = torch.from_numpy(np.frombuffer(raw, dtype=np.int32).copy()).to(torch.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")
    else:
        if sample_width == 1:
            waveform = torch.frombuffer(raw, dtype=torch.uint8).clone().to(torch.float32)
            waveform = (waveform - 128.0) / 128.0
        elif sample_width == 2:
            waveform = torch.frombuffer(raw, dtype=torch.int16).clone().to(torch.float32) / 32768.0
        elif sample_width == 4:
            waveform = torch.frombuffer(raw, dtype=torch.int32).clone().to(torch.float32) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if waveform.numel() == 0:
        waveform = torch.zeros((channels, 0), dtype=torch.float32)
    else:
        waveform = waveform.reshape(-1, channels).transpose(0, 1).contiguous()
    return waveform, sample_rate


class LTXLongAudioSegmentInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_duration": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 10_000_000.0, "step": 0.01}),
                "segment_seconds": ("INT", {"default": 20, "min": 1, "max": 3600, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 100_000, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "FLOAT", "BOOLEAN", "INT")
    RETURN_NAMES = ("start_time", "segment_seconds", "frames", "exact_seconds", "is_last_segment", "segment_count")
    FUNCTION = "segment_info"
    CATEGORY = "LTX/LongAudio"

    def segment_info(self, audio_duration, segment_seconds, index, fps):
        total_duration = max(float(audio_duration), 0.0)
        segment_seconds = max(int(segment_seconds), 1)
        index = max(int(index), 0)
        fps = max(float(fps), 1.0)
        segment_count = int(math.ceil(total_duration / segment_seconds)) if total_duration > 0 else 0
        start_time = float(index * segment_seconds)
        remaining = max(total_duration - start_time, 0.0)
        current_seconds = min(float(segment_seconds), remaining)
        is_last_segment = remaining < float(segment_seconds)
        frame_blocks = math.floor((current_seconds * fps) / 8.0) if current_seconds > 0 else 0
        frames = 1 + max(frame_blocks, 0) * 8
        exact_seconds = float((frames - 1) / fps) if frames > 1 else 0.0
        return start_time, float(current_seconds), int(frames), exact_seconds, bool(is_last_segment), segment_count


class LTXRandomImageIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_count": ("INT", {"default": 1, "min": 1, "max": 100_000, "step": 1}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100_000, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("image_index",)
    FUNCTION = "pick_index"
    CATEGORY = "LTX/LongAudio"

    def pick_index(self, image_count, segment_index, seed):
        image_count = max(int(image_count), 1)
        segment_index = max(int(segment_index), 0)
        rng = random.Random(int(seed) + (segment_index * 9973))
        return (rng.randrange(image_count),)


class CompatShowAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anything": (ANY_TYPE,)}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("output",)
    FUNCTION = "show"
    CATEGORY = "LTX/Workflow"

    def show(self, anything):
        return (anything,)


class CompatSimpleMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": ("STRING", {"default": "a + b"})},
            "optional": {"a": (ANY_TYPE,), "b": (ANY_TYPE,), "c": (ANY_TYPE,)},
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("int", "float", "boolean")
    FUNCTION = "execute"
    CATEGORY = "LTX/Workflow"

    def execute(self, value, a=0, b=0, c=0):
        result = _safe_eval(value, {"a": a, "b": b, "c": c})
        return int(result), float(result), bool(int(result))


class CompatSeedList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_num": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "max_num": ("INT", {"default": 0xFFFFFFFF, "min": 0, "max": 0xFFFFFFFF}),
                "method": (["random", "increment", "decrement"], {"default": "random"}),
                "total": ("INT", {"default": 1, "min": 1, "max": 100_000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "total")
    FUNCTION = "doit"
    CATEGORY = "LTX/Workflow"

    def doit(self, min_num, max_num, method, total, seed=0):
        min_num = int(min_num)
        max_num = int(max_num)
        total = int(total)
        rng = random.Random(int(seed))
        if min_num > max_num:
            min_num, max_num = max_num, min_num
        values = []
        for index in range(total):
            if method == "increment":
                values.append(min(max_num, min_num + index))
            elif method == "decrement":
                values.append(max(min_num, max_num - index))
            else:
                values.append(rng.randint(min_num, max_num))
        return values, total


class CompatCompare:
    OPTIONS = {
        "a == b": lambda a, b: a == b,
        "a != b": lambda a, b: a != b,
        "a < b": lambda a, b: a < b,
        "a > b": lambda a, b: a > b,
        "a <= b": lambda a, b: a <= b,
        "a >= b": lambda a, b: a >= b,
        "a > 0": lambda a, b: a > 0,
        "a <= 0": lambda a, b: a <= 0,
        "b > 0": lambda a, b: b > 0,
        "b <= 0": lambda a, b: b <= 0,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (ANY_TYPE,),
                "b": (ANY_TYPE,),
                "comparison": (list(cls.OPTIONS.keys()), {"default": "a == b"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "LTX/Workflow"

    def compare(self, a=0, b=0, comparison="a == b"):
        return (bool(self.OPTIONS[comparison](a, b)),)


class CompatIfElse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"boolean": ("BOOLEAN",), "on_true": (ANY_TYPE,), "on_false": (ANY_TYPE,)}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("*",)
    FUNCTION = "pick"
    CATEGORY = "LTX/Workflow"

    def pick(self, boolean, on_true, on_false):
        return (on_true if boolean else on_false,)


class CompatIndexAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"any": (ANY_TYPE,), "index": ("INT", {"default": 0, "min": -1_000_000, "max": 1_000_000})}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("out",)
    FUNCTION = "index"
    CATEGORY = "LTX/Workflow"

    def index(self, any, index):
        index = int(index)
        if torch is not None and isinstance(any, torch.Tensor):
            if any.shape[0] == 0:
                raise ValueError("Cannot index an empty tensor batch.")
            if index < 0:
                index += any.shape[0]
            index = min(max(index, 0), any.shape[0] - 1)
            return (any[index:index + 1].clone(),)
        if isinstance(any, (list, tuple)):
            if len(any) == 0:
                raise ValueError("Cannot index an empty list.")
            if index < 0:
                index += len(any)
            index = min(max(index, 0), len(any) - 1)
            return (any[index],)
        return (any,)


class CompatBatchAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"any_1": (ANY_TYPE,), "any_2": (ANY_TYPE,)}}

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("batch",)
    FUNCTION = "batch"
    CATEGORY = "LTX/Workflow"

    def batch(self, any_1, any_2):
        if any_1 is None:
            return (any_2,)
        if any_2 is None:
            return (any_1,)
        if torch is not None and isinstance(any_1, torch.Tensor) and isinstance(any_2, torch.Tensor):
            return (torch.cat((any_1, any_2), dim=0),)
        if isinstance(any_1, dict) and isinstance(any_2, dict) and "samples" in any_1 and "samples" in any_2:
            out = any_1.copy()
            out["samples"] = torch.cat((any_1["samples"], any_2["samples"]), dim=0)
            return (out,)
        if isinstance(any_1, tuple):
            return (any_1 + (any_2,),)
        if isinstance(any_1, list):
            return (any_1 + [any_2],)
        if isinstance(any_2, list):
            return ([any_1] + any_2,)
        return ([any_1, any_2],)


class CompatWhileLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {"condition": ("BOOLEAN", {"default": True})}, "optional": {}}
        for index in range(MAX_FLOW_NUM):
            inputs["optional"][f"initial_value{index}"] = (ANY_TYPE,)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + [ANY_TYPE] * MAX_FLOW_NUM)
    RETURN_NAMES = tuple(["flow"] + [f"value{index}" for index in range(MAX_FLOW_NUM)])
    FUNCTION = "while_loop_open"
    CATEGORY = "LTX/Workflow"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for index in range(MAX_FLOW_NUM):
            value = kwargs.get(f"initial_value{index}")
            if condition or ExecutionBlocker is None:
                values.append(value)
            else:
                values.append(ExecutionBlocker(None))
        return tuple(["stub"] + values)


class CompatWhileLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {"flow": ("FLOW_CONTROL", {"rawLink": True}), "condition": ("BOOLEAN", {})},
            "optional": {},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
        for index in range(MAX_FLOW_NUM):
            inputs["optional"][f"initial_value{index}"] = (ANY_TYPE,)
        return inputs

    RETURN_TYPES = tuple([ANY_TYPE] * MAX_FLOW_NUM)
    RETURN_NAMES = tuple([f"value{index}" for index in range(MAX_FLOW_NUM)])
    FUNCTION = "while_loop_close"
    CATEGORY = "LTX/Workflow"

    def _explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for value in node_info["inputs"].values():
            if not is_link(value):
                continue
            parent_id = value[0]
            display_id = dynprompt.get_display_node_id(parent_id)
            display_node = dynprompt.get_node(display_id)
            class_type = display_node["class_type"]
            if class_type not in ["LTXForLoopEnd", "LTXWhileLoopEnd"]:
                parent_ids.append(display_id)
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
            upstream[parent_id].append(node_id)

    def _explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, input_link in output_nodes.items():
                node_id = input_link[0]
                if node_id in parent_ids and display_id == node_id and output_id not in upstream[parent_id]:
                    if "." in parent_id:
                        parts = parent_id.split(".")
                        parts[-1] = output_id
                        upstream[parent_id].append(".".join(parts))
                    else:
                        upstream[parent_id].append(output_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            return tuple(kwargs.get(f"initial_value{index}") for index in range(MAX_FLOW_NUM))
        _require("GraphBuilder", GraphBuilder)
        if dynprompt is None or unique_id is None:
            raise RuntimeError("Loop expansion requires ComfyUI dynprompt metadata.")

        upstream = {}
        parent_ids = []
        self._explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        node_classes = _available_node_classes()
        for node_id, node in prompts.items():
            if "inputs" not in node:
                continue
            class_def = node_classes.get(node["class_type"])
            if getattr(class_def, "OUTPUT_NODE", False):
                for value in node["inputs"].values():
                    if is_link(value):
                        output_nodes[node_id] = value

        graph = GraphBuilder()
        self._explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = flow[0]
        self._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for key, value in original_node["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node)
        for index in range(MAX_FLOW_NUM):
            new_open.set_input(f"initial_value{index}", kwargs.get(f"initial_value{index}"))
        recurse = graph.lookup_node("Recurse")
        result = tuple(recurse.out(index) for index in range(MAX_FLOW_NUM))
        return {"result": result, "expand": graph.finalize()}


class CompatForLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"total": ("INT", {"default": 1, "min": 1, "max": 100_000})},
            "optional": {f"initial_value{index}": (ANY_TYPE,) for index in range(1, MAX_FLOW_NUM)},
            "hidden": {"initial_value0": (ANY_TYPE,), "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT"] + [ANY_TYPE] * (MAX_FLOW_NUM - 1))
    RETURN_NAMES = tuple(["flow", "index"] + [f"value{index}" for index in range(1, MAX_FLOW_NUM)])
    FUNCTION = "for_loop_start"
    CATEGORY = "LTX/Workflow"

    def for_loop_start(self, total, **kwargs):
        _require("GraphBuilder", GraphBuilder)
        graph = GraphBuilder()
        index = kwargs.get("initial_value0", 0)
        initial_values = {f"initial_value{num}": kwargs.get(f"initial_value{num}") for num in range(1, MAX_FLOW_NUM)}
        graph.node("LTXWhileLoopStart", condition=total, initial_value0=index, **initial_values)
        outputs = [kwargs.get(f"initial_value{num}") for num in range(1, MAX_FLOW_NUM)]
        return {"result": tuple(["stub", index] + outputs), "expand": graph.finalize()}


class CompatForLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"flow": ("FLOW_CONTROL", {"rawLink": True})},
            "optional": {f"initial_value{index}": (ANY_TYPE, {"rawLink": True}) for index in range(1, MAX_FLOW_NUM)},
            "hidden": {"dynprompt": "DYNPROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = tuple([ANY_TYPE] * (MAX_FLOW_NUM - 1))
    RETURN_NAMES = tuple([f"value{index}" for index in range(1, MAX_FLOW_NUM)])
    FUNCTION = "for_loop_end"
    CATEGORY = "LTX/Workflow"

    def for_loop_end(self, flow, dynprompt=None, **kwargs):
        _require("GraphBuilder", GraphBuilder)
        if dynprompt is None:
            raise RuntimeError("Loop expansion requires ComfyUI dynprompt metadata.")
        graph = GraphBuilder()
        while_open = flow[0]
        for_start = dynprompt.get_node(while_open)
        total = for_start["inputs"]["total"]
        sub = graph.node("LTXSimpleMath", value="a + b", a=[while_open, 1], b=1)
        cond = graph.node("LTXCompare", a=sub.out(0), b=total, comparison="a < b")
        input_values = {f"initial_value{index}": kwargs.get(f"initial_value{index}") for index in range(1, MAX_FLOW_NUM)}
        while_close = graph.node("LTXWhileLoopEnd", flow=flow, condition=cond.out(0), initial_value0=sub.out(0), **input_values)
        return {"result": tuple(while_close.out(index) for index in range(1, MAX_FLOW_NUM)), "expand": graph.finalize()}


class CompatLoadAudioUpload:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_audio_files() or [""]
        return {
            "required": {"audio": (files,)},
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10_000_000.0, "step": 0.01}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10_000_000.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    FUNCTION = "load_audio"
    CATEGORY = "LTX/Workflow"

    def load_audio(self, audio, start_time=0.0, duration=0.0):
        _require("torchaudio", torchaudio)
        path = _resolve_input_path(audio)
        try:
            if hasattr(torchaudio, "info"):
                info = torchaudio.info(path)
                sample_rate = int(info.sample_rate)
                frame_offset = max(0, int(float(start_time) * sample_rate))
                num_frames = 0 if float(duration) <= 0 else max(1, int(float(duration) * sample_rate))
                if num_frames > 0:
                    waveform, actual_sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
                else:
                    waveform, actual_sample_rate = torchaudio.load(path, frame_offset=frame_offset)
            else:
                waveform, actual_sample_rate = torchaudio.load(path)
                sample_rate = int(actual_sample_rate)
                frame_offset = max(0, int(float(start_time) * sample_rate))
                num_frames = 0 if float(duration) <= 0 else max(1, int(float(duration) * sample_rate))
                frame_end = waveform.shape[-1] if num_frames == 0 else min(waveform.shape[-1], frame_offset + num_frames)
                waveform = waveform[..., frame_offset:frame_end]
        except Exception:
            suffix = pathlib.Path(path).suffix.lower()
            if suffix != ".wav":
                raise
            sample_rate = _load_wav_fallback(path, 0, 0)[1]
            frame_offset = max(0, int(float(start_time) * sample_rate))
            num_frames = 0 if float(duration) <= 0 else max(1, int(float(duration) * sample_rate))
            waveform, actual_sample_rate = _load_wav_fallback(path, frame_offset, num_frames)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio_dict = {"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": int(actual_sample_rate)}
        loaded_duration = float(waveform.shape[-1] / max(actual_sample_rate, 1))
        return audio_dict, loaded_duration

    @classmethod
    def IS_CHANGED(cls, audio, **kwargs):
        return _hash_path(_resolve_input_path(audio))


class CompatLoadImages:
    @classmethod
    def INPUT_TYPES(cls):
        directories = _list_input_subdirectories() or [""]
        return {
            "required": {"directory": (directories,)},
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1_000_000, "step": 1}),
                "meta_batch": ("VHS_BatchManager",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "frame_count")
    FUNCTION = "load_images"
    CATEGORY = "LTX/Workflow"

    def load_images(self, directory, image_load_cap=0, skip_first_images=0, select_every_nth=1, meta_batch=None):
        _require("numpy", np)
        _require("torch", torch)
        _require("Pillow", Image)
        path = _resolve_input_path(directory)
        image_paths = []
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path) and pathlib.Path(item).suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(item_path)
        image_paths = image_paths[int(skip_first_images)::max(int(select_every_nth), 1)]
        if int(image_load_cap) > 0:
            image_paths = image_paths[:int(image_load_cap)]
        if not image_paths:
            raise FileNotFoundError(f"No images found in '{path}'.")

        frames = []
        masks = []
        expected_size = None
        for image_path in image_paths:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGBA")
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ValueError("All reference frames must share the same size.")
            array = np.asarray(image).astype("float32") / 255.0
            frames.append(torch.from_numpy(array[:, :, :3]))
            masks.append(torch.from_numpy(1.0 - array[:, :, 3]))
        return torch.stack(frames, dim=0), torch.stack(masks, dim=0), len(frames)

    @classmethod
    def IS_CHANGED(cls, directory, **kwargs):
        return _hash_path(_resolve_input_path(directory))


class CompatAudioConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "direction": (["right", "left"], {"default": "right"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "concatenate"
    CATEGORY = "LTX/Workflow"

    def concatenate(self, audio1, audio2, direction):
        _require("torch", torch)
        if int(audio1["sample_rate"]) != int(audio2["sample_rate"]):
            raise ValueError("Audio sample rates must match.")
        left = audio1["waveform"]
        right = audio2["waveform"]
        waveform = torch.cat((right, left), dim=2) if direction == "left" else torch.cat((left, right), dim=2)
        return ({"waveform": waveform, "sample_rate": int(audio1["sample_rate"])},)


class CompatIntConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 0, "min": -1_000_000, "max": 1_000_000, "step": 1})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "LTX/Workflow"

    def get_value(self, value):
        return (int(value),)


class CompatSimpleCalculatorKJ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"expression": ("STRING", {"default": "a + b", "multiline": True})},
            "optional": {
                "variables.a": ("INT,FLOAT,BOOLEAN",),
                "variables.b": ("INT,FLOAT,BOOLEAN",),
                "a": (ANY_TYPE,),
                "b": (ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("FLOAT", "INT", "BOOLEAN")
    FUNCTION = "calculate"
    CATEGORY = "LTX/Workflow"

    def calculate(self, expression, **kwargs):
        values = {}
        for key in ("variables.a", "variables.b", "a", "b"):
            if key in kwargs and kwargs[key] is not None:
                values[key.split(".")[-1]] = kwargs[key]
        result = _safe_eval(expression, values)
        return float(result), int(result), bool(result)


class CompatVAELoaderKJ:
    @classmethod
    def INPUT_TYPES(cls):
        vae_names = folder_paths.get_filename_list("vae") if folder_paths is not None else [""]
        return {
            "required": {
                "vae_name": (vae_names or [""],),
                "device": (["main_device", "offload_device"], {"default": "main_device"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "LTX/Workflow"

    def load_vae(self, vae_name, device="main_device", dtype="bf16"):
        _require("folder_paths", folder_paths)
        _require("comfy", comfy)
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state_dict = comfy.utils.load_torch_file(path)
        vae = comfy.sd.VAE(sd=state_dict)
        return (vae,)


class CompatImageResizeKJv2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"], {"default": "stretch"}),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {"mask": ("MASK",), "device": (["cpu", "gpu"],)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    FUNCTION = "resize"
    CATEGORY = "LTX/Workflow"

    def resize(self, image, width, height, upscale_method, keep_proportion, pad_color, crop_position, divisible_by, mask=None, device="cpu", unique_id=None):
        _require("torch", torch)
        if width == 0:
            width = int(image.shape[2])
        if height == 0:
            height = int(image.shape[1])
        image_work = _crop_to_aspect(image, int(width), int(height)) if keep_proportion == "crop" else image
        if int(divisible_by) > 1:
            width = max(int(divisible_by), int(width) - (int(width) % int(divisible_by)))
            height = max(int(divisible_by), int(height) - (int(height) % int(divisible_by)))
        resized = _resize_image_tensor(image_work, int(width), int(height), upscale_method).cpu()
        if mask is None:
            out_mask = torch.zeros((64, 64), dtype=torch.float32)
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask_cf = mask.unsqueeze(1) if mask.dim() == 3 else mask
            resized_mask = F.interpolate(mask_cf.float(), size=(int(height), int(width)), mode="nearest")
            out_mask = resized_mask.squeeze(1).cpu()
        return resized, int(width), int(height), out_mask


def _compute_attention_block(attention_module, query, context, transformer_options=None):
    transformer_options = transformer_options or {}
    k_value = attention_module.k_norm(attention_module.to_k(context)).to(query.dtype)
    v_value = attention_module.to_v(context).to(query.dtype)
    return comfy.ldm.modules.attention.optimized_attention(
        query,
        k_value,
        v_value,
        heads=attention_module.heads,
        attn_precision=getattr(attention_module, "attn_precision", None),
        transformer_options=transformer_options,
    ).flatten(2)


def _normalized_attention_guidance(module, positive, negative):
    guided = positive * module.nag_scale - negative * (module.nag_scale - 1)
    positive_norm = torch.norm(positive, p=1, dim=-1, keepdim=True)
    guided_norm = torch.norm(guided, p=1, dim=-1, keepdim=True)
    scale = torch.nan_to_num(guided_norm / (positive_norm + 1e-7), nan=10.0)
    adjustment = torch.where(scale > module.nag_tau, (positive_norm * module.nag_tau) / (guided_norm + 1e-7), torch.ones_like(scale))
    guided = guided * adjustment
    return guided * module.nag_alpha + positive * (1 - module.nag_alpha)


def _ltxv_cross_attention_forward_nag(module, x, context, mask=None, transformer_options=None, **kwargs):
    transformer_options = transformer_options or {}
    if context.shape[0] == 1:
        positive_x = x
        negative_x = None
        positive_context = context
        negative_context = None
    else:
        positive_x, negative_x = torch.chunk(x, 2, dim=0)
        positive_context, negative_context = torch.chunk(context, 2, dim=0)

    query_positive = module.q_norm(module.to_q(positive_x))
    positive_attention = _compute_attention_block(module, query_positive, positive_context, transformer_options)
    negative_attention = _compute_attention_block(module, query_positive, module.nag_context, transformer_options)
    output_positive = _normalized_attention_guidance(module, positive_attention, negative_attention)

    if negative_x is not None and negative_context is not None:
        query_negative = module.q_norm(module.to_q(negative_x))
        key_negative = module.k_norm(module.to_k(negative_context))
        value_negative = module.to_v(negative_context)
        output_negative = comfy.ldm.modules.attention.optimized_attention(
            query_negative,
            key_negative,
            value_negative,
            heads=module.heads,
            attn_precision=getattr(module, "attn_precision", None),
            transformer_options=transformer_options,
        )
        output = torch.cat([output_positive, output_negative], dim=0)
    else:
        output = output_positive

    if getattr(module, "to_gate_logits", None) is not None:
        gate_logits = module.to_gate_logits(x)
        batch, tokens, _ = output.shape
        output = output.view(batch, tokens, module.heads, module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        output = (output * gates.unsqueeze(-1)).view(batch, tokens, module.heads * module.dim_head)

    return module.to_out(output)


class _LTXVCrossAttentionPatch:
    def __init__(self, nag_context, nag_scale, nag_alpha, nag_tau):
        self.nag_context = nag_context
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau

    def __get__(self, obj, objtype=None):
        def wrapped(module, *args, **kwargs):
            module.nag_context = self.nag_context
            module.nag_scale = self.nag_scale
            module.nag_alpha = self.nag_alpha
            module.nag_tau = self.nag_tau
            return _ltxv_cross_attention_forward_nag(module, *args, **kwargs)

        return types.MethodType(wrapped, obj)


class CompatLTX2NAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "nag_scale": ("FLOAT", {"default": 11.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.001}),
                "nag_cond_video": ("CONDITIONING",),
                "nag_cond_audio": ("CONDITIONING",),
                "inplace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "LTX/Workflow"

    def patch(self, model, nag_scale, nag_alpha, nag_tau, nag_cond_video=None, nag_cond_audio=None, inplace=True):
        if comfy is None or mm is None or nag_scale == 0:
            return (model,)
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        dtype = model.model.manual_cast_dtype or diffusion_model.dtype
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if nag_cond_video is not None:
            context_video = nag_cond_video[0][0].to(device, dtype)
            if getattr(diffusion_model, "caption_proj_before_connector", False) and getattr(diffusion_model, "caption_projection_first_linear", False):
                diffusion_model.caption_projection.to(device)
                context_video = diffusion_model.caption_projection(context_video)
                diffusion_model.caption_projection.to(offload_device)
            if hasattr(diffusion_model, "video_embeddings_connector"):
                diffusion_model.video_embeddings_connector.to(device)
                context_video = diffusion_model.video_embeddings_connector(context_video)[0]
                diffusion_model.video_embeddings_connector.to(offload_device)
            context_video = context_video.view(1, -1, diffusion_model.inner_dim)
            for index, block in enumerate(diffusion_model.transformer_blocks):
                patch = _LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau).__get__(block.attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{index}.attn2.forward", patch)

        if nag_cond_audio is not None and getattr(diffusion_model, "audio_caption_projection", None) is not None:
            context_audio = nag_cond_audio[0][0].to(device, dtype)
            if getattr(diffusion_model, "caption_proj_before_connector", False) and getattr(diffusion_model, "caption_projection_first_linear", False):
                diffusion_model.audio_caption_projection.to(device)
                context_audio = diffusion_model.audio_caption_projection(context_audio)
                diffusion_model.audio_caption_projection.to(offload_device)
            if hasattr(diffusion_model, "audio_embeddings_connector"):
                diffusion_model.audio_embeddings_connector.to(device)
                context_audio = diffusion_model.audio_embeddings_connector(context_audio)[0]
                diffusion_model.audio_embeddings_connector.to(offload_device)
            context_audio = context_audio.view(1, -1, diffusion_model.audio_inner_dim)
            for index, block in enumerate(diffusion_model.transformer_blocks):
                patch = _LTXVCrossAttentionPatch(context_audio, nag_scale, nag_alpha, nag_tau).__get__(block.audio_attn2, block.__class__)
                model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{index}.audio_attn2.forward", patch)

        return (model_clone,)


class _LTXFFChunkPatch:
    def __init__(self, chunks, dim_threshold):
        self.chunks = int(chunks)
        self.dim_threshold = int(dim_threshold)

    def __get__(self, obj, objtype=None):
        def wrapped(module, x, *args, **kwargs):
            if x.shape[1] <= self.dim_threshold or self.chunks <= 1:
                return module.net(x)
            chunk_size = max(1, x.shape[1] // self.chunks)
            parts = []
            for index in range(self.chunks):
                start = index * chunk_size
                end = x.shape[1] if index == self.chunks - 1 else min(x.shape[1], (index + 1) * chunk_size)
                if start >= end:
                    break
                parts.append(module.net(x[:, start:end]))
            return torch.cat(parts, dim=1)

        return types.MethodType(wrapped, obj)


class CompatLTXVChunkFeedForward:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "chunks": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "dim_threshold": ("INT", {"default": 4096, "min": 0, "max": 16384, "step": 256}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "LTX/Workflow"

    def patch(self, model, chunks, dim_threshold):
        if chunks == 1:
            return (model,)
        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        for index, block in enumerate(diffusion_model.transformer_blocks):
            patch = _LTXFFChunkPatch(chunks, dim_threshold).__get__(block.ff, block.__class__)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{index}.ff.forward", patch)
        return (model_clone,)


class CompatLTX2SamplingPreviewOverride:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",), "preview_rate": ("INT", {"default": 8, "min": 1, "max": 60, "step": 1})},
            "optional": {"latent_upscale_model": ("LATENT_UPSCALE_MODEL",), "vae": ("VAE",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "LTX/Workflow"

    def patch(self, model, preview_rate, latent_upscale_model=None, vae=None):
        return (model.clone() if hasattr(model, "clone") else model,)


def _ffmpeg_executable() -> str:
    for env_name in ("LTX_FFMPEG_EXE", "IMAGEIO_FFMPEG_EXE", "FFMPEG_EXE"):
        env_value = os.environ.get(env_name)
        if env_value and os.path.exists(env_value):
            return env_value

    for candidate in ("ffmpeg", "ffmpeg.exe"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    try:
        import imageio_ffmpeg  # type: ignore

        resolved = imageio_ffmpeg.get_ffmpeg_exe()
        if resolved and os.path.exists(resolved):
            return resolved
    except Exception:
        pass

    raise RuntimeError("ffmpeg is required for video export.")


def _save_audio_wav(audio: dict[str, Any], destination: str) -> None:
    _require("torchaudio", torchaudio)
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    sample_rate = int(audio["sample_rate"])
    try:
        torchaudio.save(destination, waveform.detach().cpu(), sample_rate)
    except Exception:
        _require("torch", torch)
        waveform = waveform.detach().cpu().clamp(-1.0, 1.0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        channels = int(waveform.shape[0])
        interleaved = waveform.transpose(0, 1).contiguous().reshape(-1)
        pcm = interleaved.mul(32767.0).round().to(torch.int16).numpy().tobytes()
        with wave.open(destination, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)


def _tensor_to_uint8_frame(image_tensor) -> "Image.Image":
    _require("torch", torch)
    _require("Pillow", Image)
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    frame = image_tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(frame)


def _next_output_path(filename_prefix: str, extension: str, save_output: bool) -> tuple[str, str, str]:
    output_dir = _output_directory(save_output)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(filename_prefix).name or "LTXLongAudio"
    existing = sorted(pathlib.Path(output_dir).glob(f"{stem}_*.{extension}"))
    counter = len(existing) + 1
    file_name = f"{stem}_{counter:05d}.{extension}"
    media_type = "output" if save_output else "temp"
    return output_dir, file_name, media_type


class CompatVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "step": 1.0}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "LTX-2-long-audio"}),
                "format": (["video/h264-mp4"], {"default": "video/h264-mp4"}),
                "pix_fmt": (["yuv420p", "yuv444p"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "trim_to_audio": ("BOOLEAN", {"default": True}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {"audio": ("AUDIO",), "meta_batch": ("VHS_BatchManager",), "vae": ("VAE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    CATEGORY = "LTX/Workflow"

    def combine_video(
        self,
        images,
        frame_rate=24.0,
        loop_count=0,
        filename_prefix="LTX-2-long-audio",
        format="video/h264-mp4",
        pix_fmt="yuv420p",
        crf=19,
        save_metadata=True,
        trim_to_audio=True,
        pingpong=False,
        save_output=True,
        audio=None,
        meta_batch=None,
        vae=None,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
    ):
        _require("torch", torch)
        if isinstance(images, dict) and "samples" in images:
            images = images["samples"]
        if images is None or len(images) == 0:
            return {"ui": {"text": ["No frames to encode."]}, "result": ((save_output, []),)}

        frames = list(images)
        if pingpong and len(frames) > 2:
            frames = frames + frames[-2:0:-1]

        output_dir, file_name, media_type = _next_output_path(filename_prefix, "mp4", bool(save_output))
        output_path = os.path.join(output_dir, file_name)
        ffmpeg = _ffmpeg_executable()

        with tempfile.TemporaryDirectory(prefix="ltx-long-audio-") as temp_dir:
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for index, frame in enumerate(frames):
                _tensor_to_uint8_frame(frame).save(os.path.join(frames_dir, f"frame_{index:06d}.png"))

            command = [
                ffmpeg,
                "-y",
                "-framerate",
                str(float(frame_rate)),
                "-i",
                os.path.join(frames_dir, "frame_%06d.png"),
            ]
            if audio is not None:
                audio_path = os.path.join(temp_dir, "audio.wav")
                _save_audio_wav(audio, audio_path)
                command.extend(["-i", audio_path])

            command.extend(["-c:v", "libx264", "-pix_fmt", str(pix_fmt), "-crf", str(int(crf))])
            if audio is not None:
                command.extend(["-c:a", "aac"])
                if trim_to_audio:
                    command.append("-shortest")
            command.append(output_path)
            subprocess.run(command, check=True, capture_output=True)

        payload = {"filename": file_name, "subfolder": "", "type": media_type, "format": format}
        return {"ui": {"text": [output_path]}, "result": ((save_output, [payload]),)}


NODE_CLASS_MAPPINGS = {
    "LTXLongAudioSegmentInfo": LTXLongAudioSegmentInfo,
    "LTXRandomImageIndex": LTXRandomImageIndex,
    "LTXShowAnything": CompatShowAnything,
    "LTXSimpleMath": CompatSimpleMath,
    "LTXSeedList": CompatSeedList,
    "LTXCompare": CompatCompare,
    "LTXIfElse": CompatIfElse,
    "LTXIndexAnything": CompatIndexAnything,
    "LTXBatchAnything": CompatBatchAnything,
    "LTXWhileLoopStart": CompatWhileLoopStart,
    "LTXWhileLoopEnd": CompatWhileLoopEnd,
    "LTXForLoopStart": CompatForLoopStart,
    "LTXForLoopEnd": CompatForLoopEnd,
    "LTXLoadAudioUpload": CompatLoadAudioUpload,
    "LTXLoadImages": CompatLoadImages,
    "LTXAudioConcatenate": CompatAudioConcatenate,
    "LTXIntConstant": CompatIntConstant,
    "LTXSimpleCalculator": CompatSimpleCalculatorKJ,
    "LTXVAELoader": CompatVAELoaderKJ,
    "LTXImageResize": CompatImageResizeKJv2,
    "LTXChunkFeedForward": CompatLTXVChunkFeedForward,
    "LTXSamplingPreviewOverride": CompatLTX2SamplingPreviewOverride,
    "LTXNormalizedAttentionGuidance": CompatLTX2NAG,
    "LTXVideoCombine": CompatVideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {key: key for key in NODE_CLASS_MAPPINGS}
