from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runner

FAKE_COLAB = r'''#!PYTHON_EXECUTABLE
import json, os, pathlib, shutil, sys

args = sys.argv[1:]
if args and args[0].startswith("--auth="):
    args = args[1:]
command = args[0]
args = args[1:]

root = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
root.mkdir(parents=True, exist_ok=True)
log = root / "calls.jsonl"

def record(kind, **values):
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": kind, **values}) + "\n")

if command == "usage":
    print("Current balance: 150.00 compute units")
    print("Usage rate: 0.00/hr")
    print("Active assignments: 0")
elif command == "new":
    record("new", args=args)
    print("Session created")
elif command == "stop":
    record("stop", args=args)
    print("Session stopped")
elif command == "upload":
    remote = args[-1]
    source = pathlib.Path(args[-2])
    remote_path = root / "remote" / remote.lstrip("/")
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, remote_path)
    record("upload", remote=remote, source=source.name)
elif command == "exec":
    record("exec", args=args)
    envs = [args[i + 1] for i, value in enumerate(args[:-1]) if value == "--env"]
    env_dict = dict(item.split("=", 1) for item in envs)
    output = env_dict.get("QWEN_OUTPUT_PATH", "/content/qwen_output.png")
    remote_out = root / "remote" / output.lstrip("/")
    remote_out.parent.mkdir(parents=True, exist_ok=True)
    remote_out.write_bytes(b"fake png image bytes")
elif command == "download":
    remote = args[-2]
    target = pathlib.Path(args[-1])
    source = root / "remote" / remote.lstrip("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record("download", remote=remote, target=str(target))
else:
    print(f"Unknown fake command: {command}", file=sys.stderr)
    sys.exit(1)
'''


class QwenRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.root / "colab-state"
        self.fake_colab = self.bin_dir / "colab"

        content = FAKE_COLAB.replace("PYTHON_EXECUTABLE", sys.executable)
        self.fake_colab.write_text(content, encoding="utf-8")
        self.fake_colab.chmod(0o755)

        # Create Windows batch wrapper if on Windows
        if sys.platform == "win32":
            bat = self.bin_dir / "colab.bat"
            bat.write_text(f'@"{sys.executable}" "{self.fake_colab}" %*\n', encoding="utf-8")

        self.old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{self.bin_dir}{os.pathsep}{self.old_path}"
        os.environ["FAKE_COLAB_STATE_DIR"] = str(self.state_dir)

    def tearDown(self) -> None:
        os.environ["PATH"] = self.old_path
        os.environ.pop("FAKE_COLAB_STATE_DIR", None)
        self.temp_dir.cleanup()

    def calls(self) -> list[dict]:
        log = self.state_dir / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]

    def test_parse_usage(self) -> None:
        raw = "Current balance: 123.45 compute units\nUsage rate: 1.50/hr\nActive assignments: 1\n"
        data = runner.parse_usage(raw)
        self.assertEqual(data["balance"], 123.45)
        self.assertEqual(data["rate_per_hour"], 1.5)
        self.assertEqual(data["active_assignments"], 1)

    def test_resolve_jobs_valid_t2i_and_edit(self) -> None:
        img_path = self.root / "test_ref.png"
        img_path.write_bytes(b"dummy image")
        out_dir = self.root / "outputs"

        manifest = {
            "jobs": [
                {
                    "id": "job1",
                    "mode": "t2i",
                    "prompt": "a beautiful landscape",
                    "output_name": "landscape",
                },
                {
                    "id": "job2",
                    "mode": "edit",
                    "reference_images": [str(img_path)],
                    "prompt": "restyle <Picture 1> into watercolor",
                    "output_name": "restyle",
                },
            ]
        }
        jobs = runner.resolve_jobs(manifest, out_dir)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["mode"], "t2i")
        self.assertEqual(jobs[1]["mode"], "edit")
        self.assertEqual(len(jobs[1]["reference_images"]), 1)

    def test_resolve_jobs_rejects_more_than_10_images(self) -> None:
        imgs = []
        for i in range(11):
            p = self.root / f"ref_{i}.png"
            p.write_bytes(b"dummy")
            imgs.append(str(p))

        manifest = {
            "jobs": [
                {
                    "id": "too_many",
                    "prompt": "test",
                    "reference_images": imgs,
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "supports up to 10"):
            runner.resolve_jobs(manifest, self.root / "outputs")

    def test_resolve_jobs_checks_picture_numbers(self) -> None:
        img_path = self.root / "test_ref.png"
        img_path.write_bytes(b"dummy image")

        manifest = {
            "jobs": [
                {
                    "id": "bad_ref",
                    "reference_images": [str(img_path)],
                    "prompt": "mix <Picture 1> with <Picture 2>",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "references <Picture 2>, but only 1"):
            runner.resolve_jobs(manifest, self.root / "outputs")

    def test_run_batch_full_flow(self) -> None:
        manifest_file = self.root / "jobs.json"
        out_dir = self.root / "outputs"
        manifest_file.write_text(json.dumps({
            "jobs": [
                {
                    "id": "job_a",
                    "prompt": "sunset over mountains",
                    "output_name": "mountains",
                }
            ]
        }), encoding="utf-8")

        res = runner.run_batch(manifest_file, output_dir=out_dir, gpu="L4")
        self.assertEqual(res["status"], "completed")
        self.assertTrue((out_dir / "mountains.png").exists())
        self.assertEqual((out_dir / "mountains.png").read_bytes(), b"fake png image bytes")

        call_list = self.calls()
        self.assertEqual(sum(c["command"] == "new" for c in call_list), 1)
        self.assertEqual(sum(c["command"] == "upload" for c in call_list), 1)  # prompt only
        self.assertEqual(sum(c["command"] == "exec" for c in call_list), 1)
        self.assertEqual(sum(c["command"] == "download" for c in call_list), 1)
        self.assertEqual(sum(c["command"] == "stop" for c in call_list), 1)


if __name__ == "__main__":
    unittest.main()
