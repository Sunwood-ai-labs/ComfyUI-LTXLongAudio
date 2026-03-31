from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


CUSTOM_NODE_REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = CUSTOM_NODE_REPO / "samples" / "workflows" / "LTXLongAudio_CustomNodes_SmokeTest.json"
TRACKED_SAMPLE_AUDIO_NAME = "ltx-demo-tone.wav"
OPTIONAL_LONG_SAMPLE_AUDIO_NAME = "HOWL AT THE HAIRPIN2.wav"


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()


def discover_comfy_root() -> Path | None:
    env_root = _env_path("COMFYUI_ROOT")
    if env_root is not None:
        return env_root
    if CUSTOM_NODE_REPO.parent.name == "custom_nodes":
        return CUSTOM_NODE_REPO.parent.parent
    return None


def default_comfy_main(comfy_root: Path | None) -> Path | None:
    env_main = _env_path("COMFYUI_MAIN")
    if env_main is not None:
        return env_main
    if comfy_root is None:
        return None
    candidate = comfy_root / "main.py"
    if candidate.exists():
        return candidate
    return None


def default_comfy_python(comfy_root: Path | None) -> Path | None:
    env_python = _env_path("COMFYUI_PYTHON")
    if env_python is not None:
        return env_python
    if comfy_root is None:
        return None
    candidates = [
        comfy_root / ".venv" / "Scripts" / "python.exe",
        comfy_root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_user_directory(comfy_root: Path | None) -> Path | None:
    env_value = _env_path("COMFYUI_USER_DIRECTORY")
    if env_value is not None:
        return env_value
    if comfy_root is None:
        return None
    return comfy_root / "user"


def default_input_directory(comfy_root: Path | None) -> Path | None:
    env_value = _env_path("COMFYUI_INPUT_DIRECTORY")
    if env_value is not None:
        return env_value
    if comfy_root is None:
        return None
    return comfy_root / "input"


def default_output_directory(comfy_root: Path | None) -> Path | None:
    env_value = _env_path("COMFYUI_OUTPUT_DIRECTORY")
    if env_value is not None:
        return env_value
    if comfy_root is None:
        return None
    return comfy_root / "output"


def default_temp_directory(comfy_root: Path | None) -> Path | None:
    env_value = _env_path("COMFYUI_TEMP_DIRECTORY")
    if env_value is not None:
        return env_value
    return comfy_root


def default_database_url(user_directory: Path | None) -> str | None:
    env_value = os.environ.get("COMFYUI_DATABASE_URL")
    if env_value:
        return env_value
    if user_directory is None:
        return None
    return f"sqlite:///{user_directory.as_posix()}/comfyui_ci.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real ComfyUI /prompt smoke test from a workflow JSON.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=None,
        help="ComfyUI root. Auto-detected when the repo lives under ComfyUI/custom_nodes or via COMFYUI_ROOT.",
    )
    parser.add_argument("--comfy-main", type=Path, default=None, help="Path to ComfyUI main.py. Auto-detected from --comfy-root when possible.")
    parser.add_argument("--python-exe", type=Path, default=None, help="Path to the ComfyUI Python executable. Auto-detected from --comfy-root when possible.")
    parser.add_argument("--user-directory", type=Path, default=None, help="ComfyUI user directory. Defaults to <comfy-root>/user.")
    parser.add_argument("--input-directory", type=Path, default=None, help="ComfyUI input directory. Defaults to <comfy-root>/input.")
    parser.add_argument("--output-directory", type=Path, default=None, help="ComfyUI output directory. Defaults to <comfy-root>/output.")
    parser.add_argument("--temp-directory", type=Path, default=None, help="ComfyUI temp directory. Defaults to --comfy-root.")
    parser.add_argument("--database-url", default=None, help="Database URL. Defaults to COMFYUI_DATABASE_URL or a sqlite file under the user directory.")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=0, help="Optional fixed port. Use 0 to auto-pick a free port.")
    parser.add_argument("--ffmpeg-exe", type=Path, default=None, help="Optional explicit ffmpeg.exe path.")
    parser.add_argument(
        "--auto-fill-missing",
        action="store_true",
        help="Fill blank workflow inputs with demo assets. By default the smoke test requires valid workflow defaults.",
    )
    return parser.parse_args()


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    comfy_root = args.comfy_root or discover_comfy_root()
    comfy_main = args.comfy_main or default_comfy_main(comfy_root)
    python_exe = args.python_exe or default_comfy_python(comfy_root)
    user_directory = args.user_directory or default_user_directory(comfy_root)
    input_directory = args.input_directory or default_input_directory(comfy_root)
    output_directory = args.output_directory or default_output_directory(comfy_root)
    temp_directory = args.temp_directory or default_temp_directory(comfy_root)
    database_url = args.database_url or default_database_url(user_directory)

    missing = []
    if comfy_root is None:
        missing.append("--comfy-root")
    if comfy_main is None:
        missing.append("--comfy-main")
    if python_exe is None:
        missing.append("--python-exe")
    if user_directory is None:
        missing.append("--user-directory")
    if input_directory is None:
        missing.append("--input-directory")
    if output_directory is None:
        missing.append("--output-directory")
    if temp_directory is None:
        missing.append("--temp-directory")
    if database_url is None:
        missing.append("--database-url")

    if missing:
        raise RuntimeError(
            "Unable to resolve the ComfyUI runtime. Either clone this repo under ComfyUI/custom_nodes, "
            "set COMFYUI_ROOT (and related env vars), or pass the missing flags explicitly: "
            + ", ".join(missing)
        )

    resolved = argparse.Namespace(**vars(args))
    resolved.comfy_root = Path(comfy_root)
    resolved.comfy_main = Path(comfy_main)
    resolved.python_exe = Path(python_exe)
    resolved.user_directory = Path(user_directory)
    resolved.input_directory = Path(input_directory)
    resolved.output_directory = Path(output_directory)
    resolved.temp_directory = Path(temp_directory)
    resolved.database_url = str(database_url)
    return resolved


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def multipart_request(url: str, *, file_path: Path, timeout: float = 20.0) -> Any:
    boundary = f"----CodexBoundary{int(time.time() * 1000)}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = bytearray()

    def write_field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    write_field("overwrite", "1")
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{file_path.name}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_local_node_registry() -> dict[str, Any]:
    module_path = CUSTOM_NODE_REPO / "nodes.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.nodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, "NODE_CLASS_MAPPINGS", {})


def spec_primary_value(spec: Any) -> Any:
    if isinstance(spec, tuple) and spec:
        return spec[0]
    return spec


def spec_options(spec: Any) -> list[Any] | None:
    if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        if isinstance(options, list):
            return options
    primary = spec_primary_value(spec)
    if isinstance(primary, list):
        return primary
    return None


def spec_uses_widget(spec: Any) -> bool:
    primary = spec_primary_value(spec)
    return isinstance(primary, list) or (isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict))


def input_schema_map(node_class: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if node_class is None:
        return {}, []
    input_types = node_class.INPUT_TYPES()
    schema_map: dict[str, dict[str, Any]] = {}
    widget_order: list[str] = []
    for section_name in ("required", "optional"):
        section = input_types.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for name, spec in section.items():
            schema_map[name] = {"required": section_name == "required", "spec": spec}
            if spec_uses_widget(spec):
                widget_order.append(name)
    return schema_map, widget_order


def prompt_widget_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"__value__": value}
    return value


def workflow_to_prompt(workflow_path: Path, *, node_registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]:
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    links = {int(link[0]): link for link in workflow.get("links", []) if isinstance(link, list) and len(link) >= 6}
    prompt: dict[str, dict[str, Any]] = {}
    selected_widgets: dict[str, list[Any]] = {}

    for node in sorted(workflow.get("nodes", []), key=lambda item: int(item["id"])):
        node_id = str(node["id"])
        node_type = node["type"]
        node_class = node_registry.get(node_type)
        schema_map, widget_order = input_schema_map(node_class)
        if node_class is None:
            widget_order = [
                input_def.get("name")
                for input_def in node.get("inputs", [])
                if isinstance(input_def, dict) and input_def.get("link") is None and input_def.get("name")
            ]
        widget_values = node.get("widgets_values", [])
        widget_value_map = {
            widget_name: widget_values[index]
            for index, widget_name in enumerate(widget_order)
            if index < len(widget_values)
        }

        node_inputs: dict[str, Any] = {}
        input_defs = {input_def.get("name"): input_def for input_def in node.get("inputs", []) if isinstance(input_def, dict)}

        for widget_name in widget_order:
            input_def = input_defs.get(widget_name)
            if input_def is not None and input_def.get("link") is not None:
                link = links[int(input_def["link"])]
                node_inputs[widget_name] = [str(link[1]), int(link[2])]
            elif widget_name in widget_value_map:
                node_inputs[widget_name] = prompt_widget_value(widget_value_map[widget_name])
                selected_widgets.setdefault(widget_name, []).append(widget_value_map[widget_name])

        for input_name, input_def in input_defs.items():
            if input_name in node_inputs:
                continue
            link_id = input_def.get("link")
            if link_id is None:
                continue
            link = links[int(link_id)]
            node_inputs[input_name] = [str(link[1]), int(link[2])]

        prompt[node_id] = {
            "class_type": node_type,
            "inputs": node_inputs,
            "_meta": {"title": node.get("title") or node_type},
        }

    return prompt, selected_widgets


def wait_for_object_info(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            data = json_request("GET", f"{base_url}/object_info/LTXDummyRenderChunkSequence", timeout=5.0)
            if data.get("LTXDummyRenderChunkSequence"):
                return data
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for ComfyUI object info: {last_error}")


def wait_for_history(base_url: str, prompt_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = json_request("GET", f"{base_url}/history/{prompt_id}", timeout=5.0)
        if history:
            if isinstance(history, dict) and prompt_id in history:
                return history[prompt_id]
            if isinstance(history, dict):
                return history
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for prompt history for {prompt_id}")


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def discover_ffmpeg(python_exe: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    resolved = shutil.which("ffmpeg")
    if resolved:
        return Path(resolved)

    try:
        result = subprocess.run(  # noqa: S603
            [str(python_exe), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
            check=True,
            capture_output=True,
            text=True,
        )
        path = Path(result.stdout.strip())
        if path.exists():
            return path
    except Exception:  # noqa: BLE001
        return None
    return None


def choose_sample_audio_file() -> Path:
    preferred = CUSTOM_NODE_REPO / "samples" / "input" / OPTIONAL_LONG_SAMPLE_AUDIO_NAME
    fallback = CUSTOM_NODE_REPO / "samples" / "input" / TRACKED_SAMPLE_AUDIO_NAME
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No sample audio file found for API smoke test.")


def choose_sample_frame_files() -> list[Path]:
    preferred_dir = CUSTOM_NODE_REPO / "samples" / "input" / "frames_pool"
    fallback_dir = CUSTOM_NODE_REPO / "samples" / "input" / "demo_frames"
    target_dir = preferred_dir if preferred_dir.is_dir() else fallback_dir
    frame_files = sorted(
        [
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        ],
        key=lambda item: item.name.casefold(),
    )
    if not frame_files:
        raise FileNotFoundError("No sample frame images found for API smoke test.")
    return frame_files


def stage_default_input_assets(input_directory: Path) -> dict[str, Any]:
    input_directory.mkdir(parents=True, exist_ok=True)

    audio_source = choose_sample_audio_file()
    staged_audio = input_directory / TRACKED_SAMPLE_AUDIO_NAME
    try:
        shutil.copy2(audio_source, staged_audio)
    except PermissionError:
        if not staged_audio.exists():
            raise

    frames_dir = input_directory / "frames_pool"
    frames_dir.mkdir(parents=True, exist_ok=True)
    staged_frames = []
    for frame_path in choose_sample_frame_files():
        destination = frames_dir / frame_path.name
        try:
            shutil.copy2(frame_path, destination)
        except PermissionError:
            if not destination.exists():
                raise
        staged_frames.append(str(destination))

    return {
        "audio": str(staged_audio),
        "frames_dir": str(frames_dir),
        "frame_count": len(staged_frames),
    }


def upload_input_file(base_url: str, file_path: Path) -> str:
    response = multipart_request(f"{base_url}/upload/image", file_path=file_path, timeout=20.0)
    uploaded_name = response.get("name")
    if not uploaded_name:
        raise RuntimeError(f"Unexpected upload response for {file_path}: {response!r}")
    subfolder = response.get("subfolder", "")
    return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name


def choose_sample_directory_value(base_url: str) -> str:
    directory_info = json_request("GET", f"{base_url}/object_info/LTXLoadImages", timeout=5.0)["LTXLoadImages"]
    directory_options = directory_info["input"]["required"]["directory"][1]["options"]
    for preferred in ("frames_pool", "samples/input/frames_pool", "samples/input/demo_frames"):
        if preferred in directory_options:
            return preferred
    for option in directory_options:
        if option:
            return option
    raise RuntimeError(f"No usable frame-folder options available: {directory_options!r}")


def unwrap_prompt_value(value: Any) -> Any:
    if isinstance(value, dict) and "__value__" in value:
        return value["__value__"]
    return value


def prompt_value_list(value: Any) -> list[str]:
    raw_value = unwrap_prompt_value(value)
    if isinstance(raw_value, (list, tuple)):
        return [str(item) for item in raw_value if item]
    if isinstance(raw_value, str):
        return [raw_value] if raw_value else []
    if raw_value is None:
        return []
    return [str(raw_value)]


def apply_smoke_overrides(base_url: str, workflow_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    uploaded_audio_name: str | None = None
    uploaded_frame_names: list[str] = []
    selected_directory: str | None = None
    image_node_ids = [
        node_id
        for node_id in sorted(workflow_prompt, key=int)
        if workflow_prompt[node_id]["class_type"] == "LTXLoadImageUpload"
    ]
    batch_image_node_ids = [
        node_id
        for node_id in sorted(workflow_prompt, key=int)
        if workflow_prompt[node_id]["class_type"] == "LTXLoadImageBatchUpload"
    ]

    if image_node_ids or batch_image_node_ids:
        uploaded_frame_names = [upload_input_file(base_url, path) for path in choose_sample_frame_files()]
        for index, node_id in enumerate(image_node_ids):
            inputs = workflow_prompt[node_id]["inputs"]
            if not inputs.get("image"):
                inputs["image"] = uploaded_frame_names[index % len(uploaded_frame_names)]
        for node_id in batch_image_node_ids:
            inputs = workflow_prompt[node_id]["inputs"]
            if not prompt_value_list(inputs.get("image")):
                inputs["image"] = {"__value__": uploaded_frame_names}

    for node_id in sorted(workflow_prompt, key=int):
        node = workflow_prompt[node_id]
        inputs = node["inputs"]
        class_type = node["class_type"]

        if class_type == "LTXLoadImages" and not inputs.get("directory"):
            if selected_directory is None:
                selected_directory = choose_sample_directory_value(base_url)
            inputs["directory"] = selected_directory

        if class_type in {"LTXLoadAudioUpload", "LoadAudio"} and not inputs.get("audio"):
            if uploaded_audio_name is None:
                uploaded_audio_name = upload_input_file(base_url, choose_sample_audio_file())
            inputs["audio"] = uploaded_audio_name

    return {
        "selected_directory": selected_directory,
        "uploaded_audio": uploaded_audio_name,
        "uploaded_frames": uploaded_frame_names,
    }


def validate_workflow_defaults(workflow_prompt: dict[str, dict[str, Any]]) -> None:
    missing = []
    for node_id, node in sorted(workflow_prompt.items(), key=lambda item: int(item[0])):
        class_type = node["class_type"]
        inputs = node["inputs"]
        title = node.get("_meta", {}).get("title", class_type)
        if class_type == "LTXLoadImages" and not inputs.get("directory"):
            missing.append(f"{title} ({node_id}) is missing default 'directory'")
        if class_type in {"LTXLoadAudioUpload", "LoadAudio"} and not inputs.get("audio"):
            missing.append(f"{title} ({node_id}) is missing default 'audio'")
        if class_type == "LTXLoadImageUpload" and not inputs.get("image"):
            missing.append(f"{title} ({node_id}) is missing default 'image'")
        if class_type == "LTXLoadImageBatchUpload" and not prompt_value_list(inputs.get("image")):
            missing.append(f"{title} ({node_id}) is missing default 'image'")
    if missing:
        raise RuntimeError("Workflow defaults are incomplete:\n- " + "\n- ".join(missing))


def validate_expected_combo_options(base_url: str, workflow_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audio_node_name = "LTXLoadAudioUpload" if any(
        node["class_type"] == "LTXLoadAudioUpload" for node in workflow_prompt.values()
    ) else "LoadAudio"
    audio_info = json_request("GET", f"{base_url}/object_info/{audio_node_name}", timeout=5.0)[audio_node_name]
    image_upload_info = json_request("GET", f"{base_url}/object_info/LTXLoadImageUpload", timeout=5.0)["LTXLoadImageUpload"]
    batch_upload_used = any(node["class_type"] == "LTXLoadImageBatchUpload" for node in workflow_prompt.values())
    batch_upload_info = {}
    batch_upload_options: list[str] = []
    batch_upload_enabled = False
    batch_allow_batch = False
    if batch_upload_used:
        batch_upload_info = json_request("GET", f"{base_url}/object_info/LTXLoadImageBatchUpload", timeout=5.0).get(
            "LTXLoadImageBatchUpload", {}
        )
        if not batch_upload_info:
            raise RuntimeError("LTXLoadImageBatchUpload is not exposed in /object_info.")
        batch_upload_spec = batch_upload_info["input"]["required"]["image"]
        batch_upload_options = batch_upload_spec[1]["options"]
        batch_upload_enabled = bool(batch_upload_spec[1].get("image_upload"))
        batch_allow_batch = bool(batch_upload_spec[1].get("allow_batch"))

    audio_spec = audio_info["input"]["required"]["audio"]
    image_spec = image_upload_info["input"]["required"]["image"]
    audio_options = audio_spec[1]["options"]
    image_options = image_spec[1]["options"]
    directory_options: list[str] = []
    directory_info = {}
    directory_spec: list[Any] = ["COMBO", {"options": []}]
    try:
        directory_info = json_request("GET", f"{base_url}/object_info/LTXLoadImages", timeout=5.0).get("LTXLoadImages", {})
        directory_spec = directory_info.get("input", {}).get("required", {}).get("directory", directory_spec)
        directory_options = directory_spec[1]["options"]
    except Exception:  # noqa: BLE001
        directory_info = {}

    for node in workflow_prompt.values():
        class_type = node["class_type"]
        inputs = node["inputs"]
        if class_type in {"LTXLoadAudioUpload", "LoadAudio"}:
            audio_value = inputs.get("audio")
            if isinstance(audio_value, str) and audio_value not in audio_options:
                raise RuntimeError(f"Audio option {audio_value!r} not available: {audio_options!r}")
        if class_type == "LTXLoadImageUpload":
            image_value = inputs.get("image")
            if isinstance(image_value, str) and image_value not in image_options:
                raise RuntimeError(f"Image option {image_value!r} not available: {image_options!r}")
        if class_type == "LTXLoadImageBatchUpload":
            for image_value in prompt_value_list(inputs.get("image")):
                if image_value not in batch_upload_options:
                    raise RuntimeError(f"Batch image option {image_value!r} not available: {batch_upload_options!r}")
        if class_type == "LTXLoadImages":
            directory_value = inputs.get("directory")
            if isinstance(directory_value, str) and directory_value not in directory_options:
                raise RuntimeError(f"Directory option {directory_value!r} not available: {directory_options!r}")

    return {
        "audio_options": audio_options,
        "audio_upload_enabled": bool(audio_spec[1].get("audio_upload")),
        "image_options": image_options,
        "image_upload_enabled": bool(image_spec[1].get("image_upload")),
        "batch_image_options": batch_upload_options,
        "batch_image_upload_enabled": batch_upload_enabled,
        "batch_allow_batch": batch_allow_batch,
        "directory_options": directory_options,
        "directory_schema": directory_spec[0],
    }


def summarize_previews(history_entry: dict[str, Any], workflow_prompt: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    outputs = history_entry.get("outputs", {})
    preview_summaries: list[dict[str, Any]] = []

    for node_id, node in sorted(workflow_prompt.items(), key=lambda item: int(item[0])):
        class_type = node["class_type"]
        if class_type not in {"LTXVideoCombine", "PreviewImage"}:
            continue
        output_entry = outputs.get(node_id, {}) if isinstance(outputs, dict) else {}
        animated_value = output_entry.get("animated", False)
        if isinstance(animated_value, (list, tuple)):
            animated = bool(animated_value[0]) if animated_value else False
        else:
            animated = bool(animated_value)

        summary = {
            "node_id": node_id,
            "title": node.get("_meta", {}).get("title", "LTXVideoCombine"),
            "class_type": class_type,
            "has_images": bool(output_entry.get("images")),
            "animated": animated,
            "text": output_entry.get("text", []),
        }
        preview_summaries.append(summary)

    if not preview_summaries:
        raise RuntimeError("Smoke workflow did not include any PreviewImage or LTXVideoCombine output nodes.")

    missing_preview = [
        item
        for item in preview_summaries
        if not item["has_images"] or (item["class_type"] == "LTXVideoCombine" and not item["animated"])
    ]
    if missing_preview:
        raise RuntimeError(f"Expected preview metadata missing from output nodes: {missing_preview!r}")

    return preview_summaries


def main() -> int:
    args = resolve_runtime_args(parse_args())
    node_registry = load_local_node_registry()
    prompt, _selected_widgets = workflow_to_prompt(args.workflow, node_registry=node_registry)
    staged_assets = stage_default_input_assets(args.input_directory)

    port = int(args.port or pick_free_port())
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"ltx-comfyui-api-smoke-{port}.log"

    ffmpeg_path = discover_ffmpeg(args.python_exe, args.ffmpeg_exe)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if ffmpeg_path is not None:
        env["PATH"] = str(ffmpeg_path.parent) + os.pathsep + env.get("PATH", "")
        env["IMAGEIO_FFMPEG_EXE"] = str(ffmpeg_path)
        env["LTX_FFMPEG_EXE"] = str(ffmpeg_path)

    command = [
        str(args.python_exe),
        str(args.comfy_main),
        "--base-directory",
        str(args.comfy_root),
        "--user-directory",
        str(args.user_directory),
        "--input-directory",
        str(args.input_directory),
        "--output-directory",
        str(args.output_directory),
        "--temp-directory",
        str(args.temp_directory),
        "--database-url",
        args.database_url,
        "--cpu",
        "--disable-auto-launch",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--dont-print-server",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(args.comfy_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            wait_for_object_info(base_url, timeout=args.startup_timeout)
            if args.auto_fill_missing:
                upload_info = apply_smoke_overrides(base_url, prompt)
            else:
                validate_workflow_defaults(prompt)
                upload_info = {"selected_directory": None, "uploaded_audio": None, "uploaded_frames": []}
            combo_info = validate_expected_combo_options(base_url, prompt)

            response = json_request("POST", f"{base_url}/prompt", {"prompt": prompt}, timeout=20.0)
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"Missing prompt_id in /prompt response: {response!r}")

            history_entry = wait_for_history(base_url, prompt_id, timeout=args.execution_timeout)
            status = history_entry.get("status", {})
            status_text = status.get("status_str") or status.get("status")
            preview_summaries = summarize_previews(history_entry, prompt)

            result = {
                "workflow": str(args.workflow),
                "base_url": base_url,
                "prompt_id": prompt_id,
                "status": status_text,
                "history_keys": sorted(history_entry.keys()),
                "output_nodes": sorted(history_entry.get("outputs", {}).keys()),
                "preview_summaries": preview_summaries,
                "combo_info": combo_info,
                "upload_info": upload_info,
                "staged_assets": staged_assets,
                "used_workflow_defaults": not args.auto_fill_missing,
                "ffmpeg_path": str(ffmpeg_path) if ffmpeg_path is not None else None,
                "log_path": str(log_path),
            }
            print(json.dumps(result, ensure_ascii=True, indent=2))

            if status_text not in (None, "success", "completed"):
                raise RuntimeError(f"Prompt finished with unexpected status: {status_text!r}")
            return 0
        finally:
            terminate_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
