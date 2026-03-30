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
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_COMFY_ROOT = Path(r"D:\ComfyUI")
DEFAULT_COMFY_MAIN = Path(r"C:\Users\Aslan\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py")
DEFAULT_COMFY_PYTHON = Path(r"D:\ComfyUI\.venv\Scripts\python.exe")
DEFAULT_USER_DIR = Path(r"D:\ComfyUI\user")
DEFAULT_INPUT_DIR = Path(r"D:\ComfyUI\input")
DEFAULT_OUTPUT_DIR = Path(r"D:\ComfyUI\output")
DEFAULT_TEMP_DIR = Path(r"D:\ComfyUI")
DEFAULT_DATABASE_URL = "sqlite:///D:/ComfyUI/user/comfyui_ci.db"
DEFAULT_WORKFLOW = Path(r"D:\Prj\ComfyUI_LTX2_3_TI2V\LTXLongAudio_CustomNodes_SmokeTest.json")
CUSTOM_NODE_REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real ComfyUI /prompt smoke test from a workflow JSON.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--comfy-main", type=Path, default=DEFAULT_COMFY_MAIN)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_COMFY_PYTHON)
    parser.add_argument("--user-directory", type=Path, default=DEFAULT_USER_DIR)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-directory", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--execution-timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=0, help="Optional fixed port. Use 0 to auto-pick a free port.")
    parser.add_argument("--ffmpeg-exe", type=Path, default=None, help="Optional explicit ffmpeg.exe path.")
    return parser.parse_args()


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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def workflow_to_prompt(workflow_path: Path, *, node_registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]:
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    links = {int(link[0]): link for link in workflow.get("links", []) if isinstance(link, list) and len(link) >= 6}
    prompt: dict[str, dict[str, Any]] = {}
    selected_widgets: dict[str, list[Any]] = {}

    for node in sorted(workflow.get("nodes", []), key=lambda item: int(item["id"])):
        node_id = str(node["id"])
        node_type = node["type"]
        node_class = node_registry[node_type]
        schema_map, widget_order = input_schema_map(node_class)
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
                node_inputs[widget_name] = widget_value_map[widget_name]
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
            data = json_request("GET", f"{base_url}/object_info/LTXLoadAudioUpload", timeout=5.0)
            if data.get("LTXLoadAudioUpload"):
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
    preferred = CUSTOM_NODE_REPO / "samples" / "input" / "HOWL AT THE HAIRPIN2.wav"
    fallback = CUSTOM_NODE_REPO / "samples" / "input" / "ltx-demo-tone.wav"
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


def apply_smoke_overrides(base_url: str, workflow_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    uploaded_audio_name: str | None = None
    uploaded_frame_names: list[str] = []
    selected_directory: str | None = None
    image_node_ids = [
        node_id
        for node_id in sorted(workflow_prompt, key=int)
        if workflow_prompt[node_id]["class_type"] == "LTXLoadImageUpload"
    ]

    if image_node_ids:
        uploaded_frame_names = [upload_input_file(base_url, path) for path in choose_sample_frame_files()]
        for index, node_id in enumerate(image_node_ids):
            inputs = workflow_prompt[node_id]["inputs"]
            if not inputs.get("image"):
                inputs["image"] = uploaded_frame_names[index % len(uploaded_frame_names)]

    for node_id in sorted(workflow_prompt, key=int):
        node = workflow_prompt[node_id]
        inputs = node["inputs"]
        class_type = node["class_type"]

        if class_type == "LTXLoadImages" and not inputs.get("directory"):
            if selected_directory is None:
                selected_directory = choose_sample_directory_value(base_url)
            inputs["directory"] = selected_directory

        if class_type == "LTXLoadAudioUpload" and not inputs.get("audio"):
            if uploaded_audio_name is None:
                uploaded_audio_name = upload_input_file(base_url, choose_sample_audio_file())
            inputs["audio"] = uploaded_audio_name

    return {
        "selected_directory": selected_directory,
        "uploaded_audio": uploaded_audio_name,
        "uploaded_frames": uploaded_frame_names,
    }


def validate_expected_combo_options(base_url: str, workflow_prompt: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audio_info = json_request("GET", f"{base_url}/object_info/LTXLoadAudioUpload", timeout=5.0)["LTXLoadAudioUpload"]
    image_upload_info = json_request("GET", f"{base_url}/object_info/LTXLoadImageUpload", timeout=5.0)["LTXLoadImageUpload"]

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
        if class_type == "LTXLoadAudioUpload":
            audio_value = inputs.get("audio")
            if isinstance(audio_value, str) and audio_value not in audio_options:
                raise RuntimeError(f"Audio option {audio_value!r} not available: {audio_options!r}")
        if class_type == "LTXLoadImageUpload":
            image_value = inputs.get("image")
            if isinstance(image_value, str) and image_value not in image_options:
                raise RuntimeError(f"Image option {image_value!r} not available: {image_options!r}")
        if class_type == "LTXLoadImages":
            directory_value = inputs.get("directory")
            if isinstance(directory_value, str) and directory_value not in directory_options:
                raise RuntimeError(f"Directory option {directory_value!r} not available: {directory_options!r}")

    return {
        "audio_options": audio_options,
        "audio_upload_enabled": bool(audio_spec[1].get("audio_upload")),
        "image_options": image_options,
        "image_upload_enabled": bool(image_spec[1].get("image_upload")),
        "directory_options": directory_options,
        "directory_schema": directory_spec[0],
    }


def main() -> int:
    args = parse_args()
    node_registry = load_local_node_registry()
    prompt, _selected_widgets = workflow_to_prompt(args.workflow, node_registry=node_registry)

    port = int(args.port or pick_free_port())
    base_url = f"http://127.0.0.1:{port}"
    log_path = Path(tempfile.gettempdir()) / f"ltx-comfyui-api-smoke-{port}.log"

    ffmpeg_path = discover_ffmpeg(args.python_exe, args.ffmpeg_exe)
    env = os.environ.copy()
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
            upload_info = apply_smoke_overrides(base_url, prompt)
            combo_info = validate_expected_combo_options(base_url, prompt)

            response = json_request("POST", f"{base_url}/prompt", {"prompt": prompt}, timeout=20.0)
            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"Missing prompt_id in /prompt response: {response!r}")

            history_entry = wait_for_history(base_url, prompt_id, timeout=args.execution_timeout)
            status = history_entry.get("status", {})
            status_text = status.get("status_str") or status.get("status")

            result = {
                "workflow": str(args.workflow),
                "base_url": base_url,
                "prompt_id": prompt_id,
                "status": status_text,
                "history_keys": sorted(history_entry.keys()),
                "output_nodes": sorted(history_entry.get("outputs", {}).keys()),
                "combo_info": combo_info,
                "upload_info": upload_info,
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
