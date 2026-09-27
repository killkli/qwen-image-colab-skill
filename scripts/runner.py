#!/usr/bin/env python3
"""Run Alibaba Qwen-Image-2.1 image generation and editing jobs on Google Colab."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

SKILL_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = SKILL_DIR / "assets" / "Qwen_Image_2_1_Colab.ipynb"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PICTURE_RE = re.compile(r"<Picture\s+(\d+)>")
NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
AUTH = os.environ.get("COLAB_AUTH", "oauth2")


class ColabCommandError(RuntimeError):
    def __init__(self, label: str, returncode: int, output: list[str]):
        self.label = label
        self.returncode = returncode
        self.output = output
        tail = "\n".join(line for line in output[-12:] if line)
        super().__init__(f"{label} failed (exit {returncode})" + (f":\n{tail}" if tail else ""))


class ColabTimeoutError(TimeoutError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_output(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "").strip()


def write_progress(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_usage(output: str) -> dict[str, Any]:
    normalized = clean_output(output)

    def number(pattern: str, label: str) -> float:
        match = re.search(pattern, normalized, re.IGNORECASE | re.MULTILINE)
        if not match:
            raise ValueError(f"colab usage output did not contain {label!r}.")
        return float(match.group(1).replace(",", ""))

    balance = number(r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units\s*$", "Current balance")
    rate = number(r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/\s*hr\s*$", "Usage rate")
    assignments = number(r"^Active assignments:\s*(\d+)\s*$", "Active assignments")
    return {
        "balance": balance,
        "rate_per_hour": rate,
        "active_assignments": int(assignments),
        "checked_at": now_iso(),
    }


def _colab_path() -> str:
    path = shutil.which("colab")
    if not path:
        raise FileNotFoundError(
            "Colab CLI is not installed or is not on PATH. "
            "Install google-colab-cli and sign in with OAuth2 first."
        )
    return path


def call_colab(
    arguments: list[str],
    *,
    label: str,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
) -> str:
    command = [_colab_path(), f"--auth={AUTH}", *arguments]
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: list[str] = []
    assert child.stdout is not None
    try:
        for raw_line in child.stdout:
            line = raw_line.rstrip("\r\n")
            lines.append(line)
            if on_line:
                on_line(line)
        returncode = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        child.kill()
        child.wait()
        raise ColabTimeoutError(f"{label} timed out after {timeout:.0f}s") from exc
    finally:
        if child.stdout:
            child.stdout.close()
    if returncode != 0:
        raise ColabCommandError(label, returncode, lines)
    return "\n".join(lines)


def get_usage(timeout: float = 30.0) -> dict[str, Any]:
    output = call_colab(["usage"], label="colab usage", timeout=timeout)
    data = parse_usage(output)
    data["ok"] = True
    return data


def start_session(gpu: str = "L4", high_mem: bool = False, timeout: float = 180.0) -> str:
    session_name = f"qwen-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    args = ["new", f"--session={session_name}", f"--gpu={gpu}"]
    if high_mem:
        args.append("--high-mem")
    call_colab(args, label=f"colab new ({session_name})", timeout=timeout)
    return session_name


def stop_session(session: str, timeout: float = 60.0) -> None:
    call_colab(["stop", f"--session={session}"], label=f"colab stop ({session})", timeout=timeout)



def resolve_jobs(manifest: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest must contain a non-empty jobs list.")
    if len(jobs) > 50:
        raise ValueError("A single batch may contain at most 50 jobs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()

    for index, raw in enumerate(jobs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Job {index} must be an object.")

        job_id = str(raw.get("id") or f"job_{index:02d}").strip()
        if not job_id:
            raise ValueError(f"Job {index} has an empty id.")
        if job_id in seen_ids:
            raise ValueError(f"Job {index} duplicates job id: {job_id}")
        seen_ids.add(job_id)

        mode = str(raw.get("mode") or "t2i").strip().lower()
        if mode not in ("t2i", "edit", "inpaint"):
            raise ValueError(f"Job {job_id} has invalid mode: {mode!r}. Must be 't2i' or 'edit'.")

        prompt_str = raw.get("prompt")
        prompt_file = raw.get("prompt_file")
        if prompt_file:
            p_path = Path(prompt_file).expanduser().resolve()
            if not p_path.is_file() or p_path.stat().st_size == 0:
                raise ValueError(f"Job {job_id} prompt_file not found or empty: {prompt_file}")
            prompt = p_path.read_text(encoding="utf-8").strip()
        elif isinstance(prompt_str, str) and prompt_str.strip():
            prompt = prompt_str.strip()
        else:
            raise ValueError(f"Job {job_id} requires a non-empty prompt or prompt_file.")

        refs = raw.get("reference_images", raw.get("images", []))
        if not isinstance(refs, list):
            raise ValueError(f"Job {job_id} reference_images must be a list.")
        if len(refs) > 10:
            raise ValueError(f"Job {job_id} supports up to 10 reference images.")
        if mode == "edit" and len(refs) == 0:
            mode = "t2i"  # Auto-fallback if no images provided

        resolved_images: list[Path] = []
        for img in refs:
            ip = Path(str(img)).expanduser().resolve()
            if not ip.is_file() or ip.stat().st_size == 0:
                raise ValueError(f"Job {job_id} reference image not found or empty: {img}")
            resolved_images.append(ip)

        # Validate <Picture N> references
        num_images = len(resolved_images)
        for ref_match in PICTURE_RE.finditer(prompt):
            pic_num = int(ref_match.group(1))
            if pic_num < 1 or pic_num > num_images:
                raise ValueError(
                    f"Job {job_id} prompt references <Picture {pic_num}>, but only {num_images} images provided."
                )

        width = int(raw.get("width", 1024))
        height = int(raw.get("height", 1024))
        steps = int(raw.get("steps", 40))
        seed = int(raw.get("seed", 0))

        out_name = raw.get("output_name") or f"{job_id}_qwen"
        sanitized = NAME_RE.sub("_", str(out_name)).strip("_") or job_id
        target = (output_dir / f"{sanitized}.png").resolve()
        if target in seen_outputs:
            raise ValueError(f"Job {job_id} outputs to an already claimed path: {target}")
        seen_outputs.add(target)

        resolved.append({
            "id": job_id,
            "title": str(raw.get("title") or job_id),
            "mode": mode,
            "prompt": prompt,
            "reference_images": resolved_images,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "output_path": target,
        })

    return resolved


def run_batch(
    manifest_path: Path,
    *,
    output_dir: Path,
    gpu: str = "L4",
    high_mem: bool = False,
    session: str | None = None,
    stop_on_complete: bool = False,
    progress_path: Path | None = None,
    timeout: float = 7200.0,
) -> dict[str, Any]:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Bundled notebook not found: {NOTEBOOK}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = resolve_jobs(manifest_data, output_dir)

    owns_session = session is None
    active_session = session
    results: list[dict[str, Any]] = []

    progress_state: dict[str, Any] = {
        "status": "starting",
        "gpu": gpu,
        "total_jobs": len(jobs),
        "completed": 0,
        "failed": 0,
        "jobs": [],
    }
    write_progress(progress_path, progress_state)

    try:
        if active_session is None:
            progress_state["status"] = "allocating_session"
            write_progress(progress_path, progress_state)
            print(f"Allocating Colab session ({gpu})...")
            active_session = start_session(gpu=gpu, high_mem=high_mem)
            print(f"Colab session created: {active_session}")

        progress_state["session"] = active_session

        for job in jobs:
            job_id = job["id"]
            print(f"\n--- Running Job: {job_id} ({job['mode']}) ---")
            progress_state["current_job"] = job_id
            write_progress(progress_path, progress_state)

            # Upload reference images if any
            remote_refs: list[str] = []
            for idx, img_path in enumerate(job["reference_images"], start=1):
                remote_img = f"/content/qwen_{job_id}_ref_{idx}{img_path.suffix.lower()}"
                call_colab(
                    ["upload", "--session", active_session, str(img_path), remote_img],
                    label=f"colab upload image {idx}",
                    timeout=120.0,
                )
                remote_refs.append(remote_img)

            # Upload prompt
            prompt_tmp = output_dir / f".{job_id}_prompt.txt"
            prompt_tmp.write_text(job["prompt"], encoding="utf-8")
            remote_prompt = f"/content/qwen_{job_id}_prompt.txt"
            try:
                call_colab(
                    ["upload", "--session", active_session, str(prompt_tmp), remote_prompt],
                    label="colab upload prompt",
                    timeout=60.0,
                )
            finally:
                if prompt_tmp.exists():
                    prompt_tmp.unlink()

            remote_output = f"/content/qwen_{job_id}_out.png"
            envs = [
                f"QWEN_MODE={job['mode']}",
                f"QWEN_PROMPT_FILE={remote_prompt}",
                f"QWEN_WIDTH={job['width']}",
                f"QWEN_HEIGHT={job['height']}",
                f"QWEN_STEPS={job['steps']}",
                f"QWEN_OUTPUT_PATH={remote_output}",
            ]
            if job["seed"] > 0:
                envs.append(f"QWEN_SEED={job['seed']}")
            if remote_refs:
                envs.append(f"QWEN_REFERENCE_IMAGES={json.dumps(remote_refs)}")

            exec_args = ["exec", "--session", active_session, "--timeout", str(timeout)]
            for env in envs:
                exec_args.extend(["--env", env])
            exec_args.extend(["--file", str(NOTEBOOK)])

            print(f"Executing inference on session {active_session}...")
            call_colab(
                exec_args,
                label=f"colab exec ({job_id})",
                timeout=timeout + 60,
                on_line=lambda line: print(f"[{job_id}] {line}"),
            )

            # Download output image
            call_colab(
                ["download", "--session", active_session, remote_output, str(job["output_path"])],
                label=f"colab download ({job_id})",
                timeout=120.0,
            )

            if not job["output_path"].is_file() or job["output_path"].stat().st_size == 0:
                raise FileNotFoundError(f"Output image missing or zero size: {job['output_path']}")

            print(f"Job {job_id} succeeded: {job['output_path']} ({job['output_path'].stat().st_size} bytes)")
            results.append({"id": job_id, "status": "completed", "output": str(job["output_path"])})
            progress_state["completed"] += 1
            progress_state["jobs"].append({"id": job_id, "status": "completed"})
            write_progress(progress_path, progress_state)

        progress_state["status"] = "completed"
        write_progress(progress_path, progress_state)

    except Exception as exc:
        progress_state["status"] = "failed"
        progress_state["error"] = str(exc)
        write_progress(progress_path, progress_state)
        raise
    finally:
        if (owns_session or stop_on_complete) and active_session:
            print(f"Stopping Colab session {active_session}...")
            try:
                stop_session(active_session)
                print(f"Session {active_session} stopped cleanly.")
            except Exception as stop_exc:
                print(f"Warning: Failed to stop session {active_session}: {stop_exc}")

    return {
        "status": "completed",
        "session": active_session,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alibaba Qwen-Image-2.1 Colab runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # usage
    usage_p = subparsers.add_parser("usage", help="Check Colab balance")
    usage_p.add_argument("--json", action="store_true", help="Output JSON")

    # start
    start_p = subparsers.add_parser("start", help="Start a Colab session")
    start_p.add_argument("--gpu", default="L4", help="Colab GPU (default: L4)")
    start_p.add_argument("--high-mem", action="store_true", help="Request high RAM")

    # stop
    stop_p = subparsers.add_parser("stop", help="Stop a Colab session")
    stop_p.add_argument("--session", required=True, help="Session name")

    # batch
    batch_p = subparsers.add_parser("batch", help="Run batch jobs from manifest")
    batch_p.add_argument("--manifest", required=True, help="Path to jobs.json")
    batch_p.add_argument("--output-dir", default="./outputs", help="Output directory")
    batch_p.add_argument("--gpu", default="L4", help="Colab GPU (default: L4)")
    batch_p.add_argument("--high-mem", action="store_true", help="Request high RAM")
    batch_p.add_argument("--session", help="Reuse existing session")
    batch_p.add_argument("--stop-on-complete", action="store_true", help="Stop session after batch")
    batch_p.add_argument("--progress", help="Progress state JSON path")
    batch_p.add_argument("--timeout", type=float, default=7200.0, help="Job timeout in seconds")

    # single
    single_p = subparsers.add_parser("single", help="Run a single image generation/edit")
    single_p.add_argument("--prompt", "-p", required=True, help="Prompt text or path to prompt.txt")
    single_p.add_argument("--image", "-i", action="append", default=[], help="Reference image(s)")
    single_p.add_argument("--output", "-o", help="Target output file path")
    single_p.add_argument("--mode", choices=["t2i", "edit"], default=None, help="Generation mode")
    single_p.add_argument("--width", type=int, default=1024, help="Width (default: 1024)")
    single_p.add_argument("--height", type=int, default=1024, help="Height (default: 1024)")
    single_p.add_argument("--steps", type=int, default=40, help="Inference steps (default: 40)")
    single_p.add_argument("--gpu", default="L4", help="Colab GPU (default: L4)")
    single_p.add_argument("--timeout", type=float, default=3600.0, help="Timeout in seconds")

    args = parser.parse_args()

    if args.command == "usage":
        data = get_usage()
        if args.json:
            print(json.dumps(data))
        else:
            print(f"Current balance: {data['balance']:.2f} compute units")
            print(f"Usage rate: {data['rate_per_hour']:.2f}/hr")
            print(f"Active assignments: {data['active_assignments']}")

    elif args.command == "start":
        sess = start_session(gpu=args.gpu, high_mem=args.high_mem)
        print(f"Created session: {sess}")

    elif args.command == "stop":
        stop_session(args.session)
        print(f"Stopped session: {args.session}")

    elif args.command == "batch":
        manifest_path = Path(args.manifest).expanduser().resolve()
        out_dir = Path(args.output_dir).expanduser().resolve()
        progress_path = Path(args.progress).expanduser().resolve() if args.progress else None
        res = run_batch(
            manifest_path,
            output_dir=out_dir,
            gpu=args.gpu,
            high_mem=args.high_mem,
            session=args.session,
            stop_on_complete=args.stop_on_complete,
            progress_path=progress_path,
            timeout=args.timeout,
        )
        print("\nBatch execution completed:")
        print(json.dumps(res, indent=2))

    elif args.command == "single":
        prompt_val = args.prompt
        is_file = os.path.isfile(prompt_val)
        mode = args.mode or ("edit" if args.image else "t2i")
        output_file = Path(args.output or "output_qwen.png").expanduser().resolve()

        job_spec: dict[str, Any] = {
            "id": "single_job",
            "mode": mode,
            "reference_images": args.image,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "output_name": output_file.stem,
        }
        if is_file:
            job_spec["prompt_file"] = str(Path(prompt_val).resolve())
        else:
            job_spec["prompt"] = prompt_val

        temp_manifest = output_file.parent / ".single_manifest.json"
        temp_manifest.write_text(json.dumps({"jobs": [job_spec]}), encoding="utf-8")
        try:
            run_batch(
                temp_manifest,
                output_dir=output_file.parent,
                gpu=args.gpu,
                timeout=args.timeout,
            )
        finally:
            if temp_manifest.exists():
                temp_manifest.unlink()


if __name__ == "__main__":
    main()
