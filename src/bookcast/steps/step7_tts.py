from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from queue import Empty, Queue
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List

import soundfile as sf

from ..core.io import read_json, write_json
from ..core.runtime_log import console_print
from ..tts_priority_queue import pop_tts_priority_episode


logger = logging.getLogger("anypod.step7")
PIPELINE_TOTAL_STEPS = 8
PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR.parent
TTS_SUBPROCESS_PATH = PACKAGE_DIR / "tts_subprocess.py"
TTS_READY_MARKER = "__ANYPOD_TTS_READY__ "
TTS_RESULT_MARKER = "__ANYPOD_TTS_RESULT__ "
print = console_print
TTS_BACKEND_CONDA_ENVS = {
    "moss-ttsd": "anypod_moss_tts",
    "moss-tts": "anypod_moss_tts",
    "vibevoice": "anypod_vibevoice",
    "moss-tts(api)": "anypod",
}
_MATERIALIZING_PROGRESS_PATTERN = re.compile(r",\s*Materializing param=.*?(?=\]|\s*$)")


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def episode_index_from_id(episode_id: str) -> int:
    digits = "".join(ch for ch in episode_id if ch.isdigit())
    return int(digits) if digits else 0


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


def sanitize_tts_runtime_log_line(line: str) -> str:
    cleaned = _MATERIALIZING_PROGRESS_PATTERN.sub("", line)
    return cleaned.rstrip()


class Step7TTS:
    def __init__(
        self,
        output_dir: Path,
        episode_ids: List[str] | None = None,
        tts_backend: str = "moss-ttsd",
        tts_options: Dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.episode_ids = episode_ids or []
        self.tts_backend = ensure_string(tts_backend, "moss-ttsd")
        if self.tts_backend not in TTS_BACKEND_CONDA_ENVS:
            raise ValueError(f"Unsupported TTS backend: {self.tts_backend}")
        self.tts_options = {
            key: value
            for key, value in (tts_options or {}).items()
            if value is not None and (not isinstance(value, str) or value.strip())
        }
        self.speaker_bible_path = self.output_dir / "speaker_bible.json"
        self.script_dir = self.output_dir / "scripts"
        self.audio_dir = self.output_dir / "audios"
        self.audio_segment_dir = self.output_dir / "audio_segments"
        self.audio_meta_dir = self.output_dir / "audio_meta"
        self._tts_runtime_process: subprocess.Popen[str] | None = None

    def ensure_output_dirs(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_segment_dir.mkdir(parents=True, exist_ok=True)
        self.audio_meta_dir.mkdir(parents=True, exist_ok=True)

    def resolve_conda_env(self) -> str:
        conda_env = TTS_BACKEND_CONDA_ENVS.get(self.tts_backend)
        if conda_env is None:
            raise ValueError(f"No conda environment is configured for TTS backend {self.tts_backend}")
        return conda_env

    def build_subprocess_command(self) -> str:
        python_args = [
            "python",
            str(TTS_SUBPROCESS_PATH),
            "--output_dir",
            str(self.output_dir),
            "--tts_backend",
            self.tts_backend,
            "--serve",
        ]
        for option_name, option_value in sorted(self.tts_options.items()):
            if option_value is None:
                continue
            if isinstance(option_value, str) and not option_value.strip():
                continue
            python_args.extend([f"--{option_name}", str(option_value)])

        conda_home = os.environ.get("ANYPOD_CONDA_HOME", "")
        if not conda_home:
            raise RuntimeError(
                "Environment variable ANYPOD_CONDA_HOME is not set. "
                "Please set it to your conda installation path, e.g. export ANYPOD_CONDA_HOME=~/miniconda3"
            )
        conda_env = self.resolve_conda_env()
        activate_script = shlex.quote(str(Path(conda_home).expanduser() / "etc" / "profile.d" / "conda.sh"))
        pythonpath_prefix = shlex.quote(str(SRC_DIR))
        pythonpath_command = f"export PYTHONPATH={pythonpath_prefix}${{PYTHONPATH:+:$PYTHONPATH}}"
        python_command = shlex.join(python_args)
        return (
            f"source {activate_script} && "
            f"conda activate {conda_env} && "
            f"{pythonpath_command} && "
            f"{python_command}"
        )

    def start_tts_runtime(self) -> None:
        if self._tts_runtime_process is not None and self._tts_runtime_process.poll() is None:
            return

        command = self.build_subprocess_command()
        logger.info("Starting persistent TTS subprocess: backend=%s", self.tts_backend)
        self._tts_runtime_process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SRC_DIR.parent),
        )
        try:
            self._wait_for_runtime_ready()
        except Exception:
            self.close_tts_runtime()
            raise

    def _wait_for_runtime_ready(self) -> None:
        process = self._tts_runtime_process
        if process is None or process.stdout is None:
            raise RuntimeError("TTS subprocess did not start correctly")

        while True:
            raw_line = process.stdout.readline()
            if raw_line == "":
                return_code = process.poll()
                raise RuntimeError(f"TTS subprocess startup failed ({self.tts_backend}): exit code {return_code}")
            line = raw_line.rstrip("\n")
            if line.startswith(TTS_READY_MARKER):
                return
            if line.strip():
                line = sanitize_tts_runtime_log_line(line)
                print(f"      [TTS subprocess][init] {line}", flush=True)

    def close_tts_runtime(self) -> None:
        process = self._tts_runtime_process
        if process is None:
            return

        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(json.dumps({"action": "shutdown"}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                process.stdin.close()
        except Exception:
            pass

        remaining_output = ""
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            if process.stdout is not None:
                try:
                    remaining_output = process.stdout.read() or ""
                except Exception:
                    remaining_output = ""
                try:
                    process.stdout.close()
                except Exception:
                    pass
            self._tts_runtime_process = None

        for line in remaining_output.splitlines():
            if not line or line.startswith(TTS_RESULT_MARKER) or line.startswith(TTS_READY_MARKER):
                continue
            line = sanitize_tts_runtime_log_line(line)
            print(f"      [TTS subprocess] {line}", flush=True)

    def run_tts_subprocess(self, episode_id: str) -> Dict[str, Any]:
        self.start_tts_runtime()
        process = self._tts_runtime_process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent TTS subprocess is unavailable")
        if process.poll() is not None:
            raise RuntimeError(f"TTS subprocess exited early ({self.tts_backend}): exit code {process.returncode}")

        logger.info("Dispatching TTS job: episode=%s backend=%s", episode_id, self.tts_backend)
        process.stdin.write(
            json.dumps(
                {
                    "action": "synthesize",
                    "episode_id": episode_id,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        process.stdin.flush()

        while True:
            raw_line = process.stdout.readline()
            if raw_line == "":
                return_code = process.poll()
                raise RuntimeError(f"TTS subprocess exited unexpectedly ({self.tts_backend}): exit code {return_code}")
            line = raw_line.rstrip("\n")
            if line.startswith(TTS_RESULT_MARKER):
                payload = json.loads(line[len(TTS_RESULT_MARKER) :])
                payload_episode_id = ensure_string(payload.get("episode_id"))
                if payload_episode_id != episode_id:
                    raise RuntimeError(
                        f"TTS subprocess returned a mismatched episode result: {payload_episode_id}, expected {episode_id}"
                    )
                if ensure_string(payload.get("status")) != "ok":
                    raise RuntimeError(
                        f"TTS subprocess failed ({self.tts_backend}): "
                        f"{ensure_string(payload.get('error'), 'unknown error')}"
                    )
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("TTS subprocess returned an invalid result payload")
                return result
            if line.strip():
                line = sanitize_tts_runtime_log_line(line)
                print(f"      [TTS subprocess][{episode_id}] {line}", flush=True)

    def list_script_files(self) -> List[Path]:
        if not self.script_dir.exists():
            return []
        script_files = sorted(path for path in self.script_dir.glob("*.txt") if path.is_file())
        if not self.episode_ids:
            return script_files
        wanted = {episode_id.strip() for episode_id in self.episode_ids if episode_id.strip()}
        return [path for path in script_files if path.stem in wanted]

    def prepare_runtime(self) -> Dict[str, Any]:
        if not self.speaker_bible_path.exists():
            raise FileNotFoundError(f"Missing speaker bible input: {self.speaker_bible_path}")
        if not TTS_SUBPROCESS_PATH.exists():
            raise FileNotFoundError(f"Missing TTS subprocess entrypoint: {TTS_SUBPROCESS_PATH}")

        self.ensure_output_dirs()
        speaker_bible = read_json(self.speaker_bible_path)
        self.start_tts_runtime()
        return speaker_bible

    def build_audio_meta(
        self,
        episode_id: str,
        script_path: Path,
        output_path: Path,
        segment_paths: List[str],
        speaker_bible: Dict[str, Any],
        tts_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        info = sf.info(str(output_path))
        return {
            "episode_id": episode_id,
            "script_path": str(script_path),
            "output_path": str(output_path),
            "segment_paths": segment_paths,
            "sample_rate": int(info.samplerate),
            "frame_count": int(info.frames),
            "channel_count": int(info.channels),
            "actual_duration_sec": float(info.duration),
            "script_char_count": len(script_path.read_text(encoding="utf-8")),
            "tts_voice_ids": [
                ensure_string(speaker.get("tts_voice_id"))
                for speaker in speaker_bible.get("speakers", [])
                if isinstance(speaker, dict)
            ],
            "tts_backend": self.tts_backend,
            "tts_mode": ensure_string(tts_result.get("tts_mode"), self.tts_backend),
            "tts_module_name": ensure_string(tts_result.get("tts_module_name"), self.tts_backend),
            "tts_conda_env": self.resolve_conda_env(),
        }

    def synthesize_episode(
        self,
        episode_id: str,
        speaker_bible: Dict[str, Any],
    ) -> Dict[str, Any]:
        script_path = self.script_dir / f"{episode_id}.txt"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script file: {script_path}")

        result = self.run_tts_subprocess(episode_id=episode_id)
        output_path = Path(
            ensure_string(result.get("output_path"), str(self.audio_dir / f"{episode_id}.wav"))
        )
        raw_segment_paths = result.get("segment_paths")
        if not isinstance(raw_segment_paths, list):
            raw_segment_paths = [str(output_path)]
        segment_paths = [ensure_string(item) for item in raw_segment_paths if ensure_string(item)]

        audio_meta = self.build_audio_meta(
            episode_id=episode_id,
            script_path=script_path,
            output_path=output_path,
            segment_paths=segment_paths,
            speaker_bible=speaker_bible,
            tts_result=result,
        )
        audio_meta_path = self.audio_meta_dir / f"{episode_id}.json"
        write_json(audio_meta, audio_meta_path)
        return {
            "episode_id": episode_id,
            "script_path": str(script_path),
            "output_path": str(output_path),
            "audio_meta_path": str(audio_meta_path),
            "segment_paths": segment_paths,
        }

    def build_result(
        self,
        generated_episode_ids: List[str],
        failed_episode_ids: List[str] | None = None,
        error_messages: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "audio_dir": str(self.audio_dir),
            "audio_segment_dir": str(self.audio_segment_dir),
            "audio_meta_dir": str(self.audio_meta_dir),
            "generated_episode_ids": generated_episode_ids,
            "failed_episode_ids": failed_episode_ids or [],
            "error_messages": error_messages or {},
        }

    def run(self) -> Dict[str, Any]:
        if not self.script_dir.exists():
            raise FileNotFoundError(f"Missing script directory: {self.script_dir}")
        try:
            speaker_bible = self.prepare_runtime()
            script_files = self.list_script_files()
            if not script_files:
                raise FileNotFoundError(f"No usable scripts were found: {self.script_dir}")

            generated_episode_ids: List[str] = []
            for script_path in sorted(script_files, key=lambda path: episode_index_from_id(path.stem)):
                episode_result = self.synthesize_episode(
                    episode_id=script_path.stem,
                    speaker_bible=speaker_bible,
                )
                generated_episode_ids.append(episode_result["episode_id"])

            return self.build_result(generated_episode_ids=generated_episode_ids)
        finally:
            self.close_tts_runtime()


class AsyncTTSWorker:
    def __init__(self, step7_tts: Step7TTS, total_episode_count: int = 0) -> None:
        self.step7_tts = step7_tts
        self.task_queue: Queue[str | None] = Queue()
        self.generated_episode_ids: List[str] = []
        self.failed_episode_ids: List[str] = []
        self.error_messages: Dict[str, str] = {}
        self._speaker_bible: Dict[str, Any] | None = None
        self._worker_thread: Thread | None = None
        self.total_episode_count = max(int(total_episode_count), 0)
        self.started_episode_count = 0

    def build_progress_label(self, current_index: int) -> str:
        if self.total_episode_count > 0 and current_index <= self.total_episode_count:
            return f"{current_index}/{self.total_episode_count}"
        return str(current_index)

    def start(self) -> None:
        self._speaker_bible = self.step7_tts.prepare_runtime()
        self._worker_thread = Thread(
            target=self._run_loop,
            name="anypod_tts_worker",
            daemon=False,
        )
        self._worker_thread.start()

    def enqueue_episode(self, episode_id: str) -> None:
        self.task_queue.put(episode_id)

    def close_and_wait(self) -> Dict[str, Any]:
        if self._worker_thread is None:
            self.step7_tts.close_tts_runtime()
            return self.step7_tts.build_result(
                generated_episode_ids=[],
                failed_episode_ids=[],
                error_messages={},
            )
        try:
            self.task_queue.put(None)
            self._worker_thread.join()
            return self.step7_tts.build_result(
                generated_episode_ids=self.generated_episode_ids,
                failed_episode_ids=self.failed_episode_ids,
                error_messages=self.error_messages,
            )
        finally:
            self.step7_tts.close_tts_runtime()

    def _run_loop(self) -> None:
        assert self._speaker_bible is not None

        while True:
            queue_item: str | None = None
            task_from_main_queue = False
            is_priority_task = False

            priority_episode_id = pop_tts_priority_episode(self.step7_tts.output_dir)
            if priority_episode_id:
                episode_id = priority_episode_id
                is_priority_task = True
            else:
                try:
                    queue_item = self.task_queue.get(timeout=0.5)
                except Empty:
                    continue
                task_from_main_queue = True
                if queue_item is None:
                    self.task_queue.task_done()
                    break
                episode_id = queue_item

            try:
                self.started_episode_count += 1
                progress_label = self.build_progress_label(self.started_episode_count)
                t0 = time.time()
                progress_suffix = "(priority regeneration)" if is_priority_task else f"({progress_label})"
                print(
                    f"    [Step 7/{PIPELINE_TOTAL_STEPS}] Start TTS: {episode_id} {progress_suffix}",
                    flush=True,
                )
                logger.info("Starting %sTTS: %s", "priority regeneration " if is_priority_task else "sequential ", episode_id)
                episode_result = self.step7_tts.synthesize_episode(
                    episode_id=episode_id,
                    speaker_bible=self._speaker_bible,
                )
                self.generated_episode_ids.append(episode_result["episode_id"])
                print(
                    f"    [Step 7/{PIPELINE_TOTAL_STEPS}] Finished TTS: {episode_id} "
                    f"({('priority regeneration' if is_priority_task else progress_label)}, {fmt_elapsed(time.time() - t0)})",
                    flush=True,
                )
            except Exception as exc:
                logger.warning("TTS generation failed: %s, error: %s", episode_id, exc)
                self.failed_episode_ids.append(episode_id)
                self.error_messages[episode_id] = str(exc)
                print(
                    f"    [Step 7/{PIPELINE_TOTAL_STEPS}] Failed TTS: {episode_id} "
                    f"({('priority regeneration' if is_priority_task else progress_label)}) error: {exc}",
                    flush=True,
                )
            finally:
                if task_from_main_queue:
                    self.task_queue.task_done()
