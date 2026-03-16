from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import gradio as gr

from bookcast.cli import build_parser
from bookcast.core.common import build_book_id
from bookcast.core.common import save_text
from bookcast.core.voice_library import find_voice_entry
from bookcast.core.voice_library import load_voice_library
from bookcast.core.voice_library import upsert_voice_entry
from bookcast.steps.step7_tts import Step7TTS
from bookcast.tts_priority_queue import clear_tts_priority_queue
from bookcast.tts_priority_queue import enqueue_tts_priority_episode
from bookcast.tts_text import normalize_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gradio_outputs"
DEFAULT_ARGS = build_parser().parse_args([])
APP_THEME = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate")
ALLOWED_RETURN_ROOTS = (REPO_ROOT.resolve(), DEFAULT_OUTPUT_ROOT.resolve())
POLL_INTERVAL_SEC = 0.5
GRADIO_DEFAULT_STEP2_NUM_WORKERS = 8
GRADIO_DEFAULT_MAX_RETRIES = 8
DIALOGUE_MODE_LABEL_TO_CODE = {
    "单人": "single",
    "双人": "dual",
    "Single": "single",
    "Dual": "dual",
}
DIALOGUE_MODE_CODE_TO_LABEL = {code: label for label, code in DIALOGUE_MODE_LABEL_TO_CODE.items()}
PROGRAM_MODE_LABEL_TO_CODE = {
    "忠于原文": "faithful",
    "精炼总结": "concise",
    "深度讲解": "deep_dive",
    "Faithful to source": "faithful",
    "Concise summary": "concise",
    "Deep explanation": "deep_dive",
}
PROGRAM_MODE_CODE_TO_LABEL = {code: label for label, code in PROGRAM_MODE_LABEL_TO_CODE.items()}
LANGUAGE_LABEL_TO_CODE = {
    "中文": "zh",
    "英文": "en",
    "Chinese": "zh",
    "English": "en",
}
LANGUAGE_CODE_TO_LABEL = {code: label for label, code in LANGUAGE_LABEL_TO_CODE.items()}
SCRIPT_TURN_TAG_PATTERN = re.compile(r"\[S\d+\]")

UI_LANGUAGES = ("en", "zh")
UI_LANGUAGE_LABELS = {
    "en": {"en": "English", "zh": "Chinese"},
    "zh": {"en": "英文", "zh": "中文"},
}
DIALOGUE_MODE_UI_LABELS = {
    "en": {"single": "Single", "dual": "Dual"},
    "zh": {"single": "单人", "dual": "双人"},
}
PROGRAM_MODE_UI_LABELS = {
    "en": {
        "faithful": "Faithful to source",
        "concise": "Concise summary",
        "deep_dive": "Deep explanation",
    },
    "zh": {
        "faithful": "忠于原文",
        "concise": "精炼总结",
        "deep_dive": "深度讲解",
    },
}
CONTENT_LANGUAGE_UI_LABELS = {
    "en": {"zh": "Chinese", "en": "English"},
    "zh": {"zh": "中文", "en": "英文"},
}

UI_TEXTS = {
    "en": {
        "app_intro": "# AnyPod Studio",
        "ui_language_label": "Interface language",
        "settings_title": "## Run settings",
        "run_button": "Run",
        "stop_button": "Stop current run",
        "input_section": "### Input text file",
        "uploaded_file_label": "Upload input text file",
        "input_path_label": "Or enter an absolute input path",
        "input_path_placeholder": "/absolute/path/your_input.txt or /absolute/path/your_input.pdf",
        "output_dir_label": "Absolute output directory",
        "output_dir_placeholder": "Leave empty to use gradio_outputs/ inside the repo",
        "dialogue_mode_label": "Number of hosts",
        "primary_language_label": "Primary language",
        "program_mode_label": "Program mode",
        "target_episode_level_label": "Episode length level",
        "tts_backend_label": "TTS backend",
        "allow_external_knowledge_label": "Allow external knowledge",
        "show_title_label": "Show title preference",
        "show_title_placeholder": "Example: Open Moss",
        "positioning_label": "Show positioning preference",
        "positioning_placeholder": "Example: A podcast covering daily AI news",
        "target_audience_label": "Target audience preference",
        "target_audience_placeholder": "Example: AI practitioners and tech enthusiasts with some background knowledge",
        "fixed_opening_label": "Fixed opening preference",
        "fixed_opening_placeholder": "Example: [S1]Welcome to Open Moss. I'm Moss. [S2]Hi everyone, I'm Orb.",
        "speaker_1_title": "### Host 1",
        "speaker_2_title": "### Host 2",
        "name_label": "Name",
        "speaker_1_name_placeholder": "Example: Moss",
        "speaker_2_name_placeholder": "Example: Orb",
        "style_label": "Style",
        "speaker_1_style_placeholder": "Example: Energetic, lively, and quick-thinking host",
        "speaker_2_style_placeholder": "Example: Calm, highly knowledgeable, and good at explaining",
        "voice_id_label": "Voice ID",
        "voice_library_title": "### Voice library",
        "voice_library_selector_label": "Saved voices",
        "voice_library_speaker_name_label": "Speaker name",
        "voice_library_prompt_audio_path_label": "Prompt audio path",
        "voice_library_prompt_audio_upload_label": "Upload prompt audio",
        "voice_library_prompt_text_label": "Prompt text",
        "or_separator": "or",
        "voice_library_new_button": "New voice",
        "voice_library_save_button": "Save voice",
        "voice_library_status_idle": "Select an existing voice to edit it, or create a new one below.",
        "voice_library_loaded": "Loaded voice `{voice_id}`.",
        "voice_library_saved": "Saved voice `{voice_id}`.",
        "voice_library_save_failed": "Failed to save voice: `{error}`",
        "language_label": "Language",
        "advanced_options_label": "Advanced options",
        "status_panel_title": "## Run status",
        "status_wait_title": "### Idle",
        "status_wait_detail": "Fill in the fields and click Run.",
        "status_running_title": "### Running",
        "status_terminated_title": "### Stopped",
        "status_failed_title": "### Failed",
        "status_completed_title": "### Completed",
        "status_output_dir": "Output directory",
        "logs_label": "Live logs",
        "artifact_panel_title": "## Outputs",
        "artifact_hint": "Click a filename to open the preview on the right. Audio entries can be regenerated directly.",
        "scripts_block_title": "### Generated scripts",
        "audios_block_title": "### Generated audio",
        "preview_panel_title": "## File preview",
        "preview_default_title": "### File preview",
        "preview_download_button": "Download current file",
        "preview_close_button": "Close preview",
        "audio_preview_title": "Audio preview",
        "script_editor_title": "Script editor",
        "preview_script_empty": "No script available for preview.",
        "preview_click_hint": "Click a file on the left to open an editable preview here.",
        "preview_no_turns": "There are no turns yet. Insert the first turn below.",
        "insert_first_turn": "Insert first turn",
        "insert_turn": "Insert",
        "delete_turn": "Delete",
        "append_turn": "Add a turn at the end",
        "save_script": "Save script changes",
        "no_generated_scripts": "No generated scripts yet.",
        "no_generated_audio": "No generated audio yet.",
        "download_button": "Download",
        "upload_override_button": "Upload override",
        "regenerate_button": "Regenerate",
        "host_1_fallback": "Host 1",
        "host_2_fallback": "Host 2",
        "editable_script_missing": "No editable script is available.",
        "script_file_missing": "Script file does not exist: {path}",
        "script_read_failed": "Failed to read the script: {error}",
        "script_empty": "The script is empty. Add a turn below.",
        "save_requires_selection": "Select a script or audio entry on the left before saving.",
        "save_missing_path": "The script file does not exist: {path}",
        "save_empty_rows": "Keep at least one valid turn before saving.",
        "save_success": "Saved to `{path}`.",
        "target_script_missing": "The target script does not exist and cannot be overwritten.",
        "upload_missing_file": "No uploaded file was received. Please try again.",
        "upload_only_txt": "Only .txt files can overwrite the current script.",
        "upload_normalized_empty": "The uploaded text became empty after normalization and cannot overwrite the script.",
        "upload_failed": "Overwrite failed: {error}",
        "upload_success": "Replaced `{name}` with the uploaded text file.",
        "run_validation_error_title": "### Invalid parameters",
        "run_validation_error_detail": "Please fix the input parameters: `{error}`",
        "run_launch_error_title": "### Failed to start",
        "run_launch_error_detail": "Could not start the job process: `{error}`",
        "run_terminating_detail": "Stopping the current run and waiting for the process to exit...",
        "run_running_detail": "The pipeline is running. Logs refresh on the polling timer.",
        "run_stopped_detail": "The current run was stopped manually.",
        "run_failed_detail": "Execution interrupted: `{error}`",
        "run_completed_detail": "Generation finished. The scripts and audio files are organized below.",
        "stop_no_job_title": "### Idle",
        "stop_no_job_detail": "There is no running job to stop.",
        "regen_no_job_title": "### Idle",
        "regen_no_job_detail": "No active job context is available, so audio cannot be regenerated.",
        "local_regen_detail": "Regenerating audio from the latest script with backend `{tts_backend}`.",
        "local_regen_current": "Current episode: `{episode_id}`.",
        "local_regen_pending": "Queued episodes: `{episode_ids}`.",
        "audio_player_play": "Play",
        "audio_player_pause": "Pause",
        "audio_player_empty": "No audio is available for preview.",
        "audio_player_preparing": "Preparing audio preview...",
        "audio_player_loading": "Loading the full waveform...",
        "audio_player_ready": "Full waveform loaded. Click the waveform to seek.",
        "audio_player_wave_fail": "Waveform preview failed to load. Switched to the native audio player.",
        "audio_player_lib_fail": "Waveform library failed to load. Switched to the native audio player.",
    },
    "zh": {
        "app_intro": "# AnyPod Studio",
        "ui_language_label": "界面语言",
        "settings_title": "## 任务设置",
        "run_button": "开始生成",
        "stop_button": "终止本次生成",
        "input_section": "### 输入文本文件",
        "uploaded_file_label": "上传输入文本文件",
        "input_path_label": "或填写输入文件绝对路径",
        "input_path_placeholder": "/绝对路径/your_input.txt 或 /绝对路径/your_input.pdf",
        "output_dir_label": "输出目录绝对路径",
        "output_dir_placeholder": "留空时自动写入仓库下的 gradio_outputs/",
        "dialogue_mode_label": "主持人数量",
        "primary_language_label": "主语言",
        "program_mode_label": "节目模式",
        "target_episode_level_label": "每集时长等级",
        "tts_backend_label": "TTS 后端",
        "allow_external_knowledge_label": "允许外部知识",
        "show_title_label": "节目名称偏好",
        "show_title_placeholder": "例：开放苔藓",
        "positioning_label": "节目定位偏好",
        "positioning_placeholder": "例：播报每日人工智能新闻的播客",
        "target_audience_label": "目标听众偏好",
        "target_audience_placeholder": "例：针对人工智能行业的从业者和对人工智能感兴趣且有一定背景知识的科技爱好者",
        "fixed_opening_label": "固定开场白偏好",
        "fixed_opening_placeholder": "例：[S1]欢迎来到《开放苔藓》，大家好，我是小苔藓。[S2]大家好，我是小皮球。",
        "speaker_1_title": "### 主持人一",
        "speaker_2_title": "### 主持人二",
        "name_label": "名称",
        "speaker_1_name_placeholder": "例：小苔藓",
        "speaker_2_name_placeholder": "例：小皮球",
        "style_label": "风格",
        "speaker_1_style_placeholder": "例：性格活泼，思维活跃的主持人",
        "speaker_2_style_placeholder": "例：性格沉稳，专业知识深厚，善于解释问题",
        "voice_id_label": "音色 ID",
        "voice_library_title": "### 音色库",
        "voice_library_selector_label": "已保存音色",
        "voice_library_speaker_name_label": "说话人名称",
        "voice_library_prompt_audio_path_label": "参考音频路径",
        "voice_library_prompt_audio_upload_label": "上传参考音频",
        "voice_library_prompt_text_label": "参考文本",
        "or_separator": "或",
        "voice_library_new_button": "新建音色",
        "voice_library_save_button": "保存音色",
        "voice_library_status_idle": "可先选择已有音色进行编辑，或直接在下方新建一个音色。",
        "voice_library_loaded": "已加载音色 `{voice_id}`。",
        "voice_library_saved": "已保存音色 `{voice_id}`。",
        "voice_library_save_failed": "保存音色失败：`{error}`",
        "language_label": "语言",
        "advanced_options_label": "高级选项",
        "status_panel_title": "## 运行状态",
        "status_wait_title": "### 等待运行",
        "status_wait_detail": "填好参数后点击开始。",
        "status_running_title": "### 运行中",
        "status_terminated_title": "### 已终止",
        "status_failed_title": "### 运行失败",
        "status_completed_title": "### 运行完成",
        "status_output_dir": "输出目录",
        "logs_label": "实时日志",
        "artifact_panel_title": "## 产物列表",
        "artifact_hint": "点击文件名可在右侧展开预览；音频支持直接重生成。",
        "scripts_block_title": "### 生成剧本",
        "audios_block_title": "### 生成音频",
        "preview_panel_title": "## 文件预览",
        "preview_default_title": "### 文件预览",
        "preview_download_button": "下载当前文件",
        "preview_close_button": "收起预览",
        "audio_preview_title": "音频预览",
        "script_editor_title": "剧本编辑",
        "preview_script_empty": "暂无可预览的剧本。",
        "preview_click_hint": "点击左侧文件后，这里会显示可编辑预览。",
        "preview_no_turns": "当前没有轮次，可以点击下方插入第一轮。",
        "insert_first_turn": "插入第一轮",
        "insert_turn": "插入",
        "delete_turn": "删除",
        "append_turn": "在末尾新增一轮",
        "save_script": "保存剧本修改",
        "no_generated_scripts": "暂无已生成剧本。",
        "no_generated_audio": "暂无已生成音频。",
        "download_button": "下载",
        "upload_override_button": "上传覆盖",
        "regenerate_button": "重新生成",
        "host_1_fallback": "主持人一",
        "host_2_fallback": "主持人二",
        "editable_script_missing": "暂无可编辑的剧本。",
        "script_file_missing": "剧本文件不存在：{path}",
        "script_read_failed": "读取剧本失败：{error}",
        "script_empty": "剧本为空，可点击下方新增一轮。",
        "save_requires_selection": "请先在左侧点击一个剧本或音频，再保存。",
        "save_missing_path": "剧本文件不存在：{path}",
        "save_empty_rows": "至少保留一轮有效发言后才能保存。",
        "save_success": "已保存到 `{path}`。",
        "target_script_missing": "目标剧本不存在，无法覆盖。",
        "upload_missing_file": "未读取到上传文件，请重试。",
        "upload_only_txt": "只支持上传 .txt 文件覆盖当前剧本。",
        "upload_normalized_empty": "上传文件在规范化后为空，无法覆盖当前剧本。",
        "upload_failed": "覆盖失败：{error}",
        "upload_success": "已用上传文件覆盖 `{name}`。",
        "run_validation_error_title": "### 参数错误",
        "run_validation_error_detail": "请先修正输入参数：`{error}`",
        "run_launch_error_title": "### 启动失败",
        "run_launch_error_detail": "未能启动任务进程：`{error}`",
        "run_terminating_detail": "正在终止本次生成，等待任务进程退出...",
        "run_running_detail": "流水线正在执行，日志会按定时轮询刷新。",
        "run_stopped_detail": "本次生成已被手动终止。",
        "run_failed_detail": "执行中断：`{error}`",
        "run_completed_detail": "生成结束，剧本与音频文件已整理在下方。",
        "stop_no_job_title": "### 等待运行",
        "stop_no_job_detail": "当前没有可终止的任务。",
        "regen_no_job_title": "### 等待运行",
        "regen_no_job_detail": "当前没有可用的任务上下文，无法重生成音频。",
        "local_regen_detail": "正在根据最新剧本重生成音频，后端：`{tts_backend}`。",
        "local_regen_current": "当前分集：`{episode_id}`。",
        "local_regen_pending": "排队分集：`{episode_ids}`。",
        "audio_player_play": "播放",
        "audio_player_pause": "暂停",
        "audio_player_empty": "暂无音频可预览。",
        "audio_player_preparing": "正在准备音频预览...",
        "audio_player_loading": "正在加载完整波形...",
        "audio_player_ready": "完整波形已加载，可点击波形跳转进度。",
        "audio_player_wave_fail": "波形加载失败，已切换为原生音频播放器。",
        "audio_player_lib_fail": "波形库加载失败，已切换为原生音频播放器。",
    },
}

CUSTOM_CSS = """
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  background:
    radial-gradient(circle at top left, rgba(14, 116, 144, 0.10), transparent 30%),
    radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 26%),
    linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}

.block-title {
  margin: 0 0 0.18rem;
}

.app-toolbar {
  align-items: flex-start;
  margin-bottom: 0.4rem;
}

.language-switch-wrap {
  display: flex;
  justify-content: flex-end;
  align-items: flex-start;
}

.language-switch-wrap > div,
.language-switch-wrap > div > div {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  min-height: auto !important;
  overflow: visible !important;
}

#ui-language-switch,
.language-switch-wrap > .wrap,
.language-switch-wrap .form,
.language-switch-wrap .block {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  min-height: auto !important;
  overflow: visible !important;
}

#ui-language-switch {
  min-width: auto !important;
  width: auto !important;
  display: inline-flex !important;
  justify-content: flex-end !important;
}

#ui-language-switch > .wrap,
#ui-language-switch .form {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  overflow: visible !important;
}

#ui-language-switch fieldset {
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
  min-width: auto !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}

#ui-language-switch .wrap {
  display: flex !important;
  justify-content: flex-end;
  align-items: center;
  gap: 0 !important;
}

#ui-language-switch label {
  display: inline-flex !important;
  align-items: center;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-width: auto !important;
  min-height: auto !important;
  line-height: 1 !important;
  cursor: pointer;
}

#ui-language-switch label + label {
  margin-left: 0.45rem;
  position: relative;
}

#ui-language-switch label + label::before {
  content: "|";
  position: absolute;
  left: -0.28rem;
  top: 50%;
  transform: translateY(-50%);
  color: #94a3b8;
  font-weight: 500;
}

#ui-language-switch label span {
  font-size: 0.86rem;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: #64748b;
}

#ui-language-switch label[data-selected="true"] span {
  color: #0f172a;
}

#ui-language-switch input {
  display: none !important;
}

.panel {
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
  padding: 0.7rem 0.75rem !important;
}

.run-button {
  min-height: 42px;
}

.panel > .wrap {
  gap: 0.45rem;
}

.artifact-panel,
.preview-panel {
  min-height: 0;
}

.artifact-panel > .wrap,
.preview-panel > .wrap {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.artifact-entry {
  align-items: center;
  gap: 0.36rem;
  padding: 0.28rem 0.32rem;
  border: 1px solid rgba(15, 23, 42, 0.07);
  border-radius: 10px;
  background: rgba(248, 250, 252, 0.92);
  margin-bottom: 0.24rem;
}

.artifact-file-button button {
  justify-content: flex-start !important;
  text-align: left !important;
  font-weight: 600;
  border-radius: 9px !important;
  min-height: 34px;
  padding: 0.3rem 0.55rem !important;
}

.artifact-action-button button {
  min-width: 3.9rem;
  border-radius: 9px !important;
  min-height: 34px;
  padding: 0.3rem 0.55rem !important;
}

.artifact-block,
.artifact-list-scroll,
.preview-scroll {
  min-height: 0;
}

.artifact-list-scroll,
.preview-scroll {
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 0.2rem;
}

.artifact-block {
  display: flex;
  flex: 0 0 auto;
  flex-direction: column;
  gap: 0.2rem;
}

.artifact-block-title {
  margin: 0 !important;
}

.artifact-list-scroll {
  height: calc((100vh - 300px) / 2);
  min-height: 180px;
  border: 1px solid rgba(15, 23, 42, 0.07);
  border-radius: 12px;
  background: rgba(248, 250, 252, 0.88);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
}

.artifact-list-scroll > .wrap {
  min-height: 100%;
  padding: 0.45rem 0.5rem !important;
  gap: 0.28rem;
}

.artifact-list-scroll .prose,
.preview-scroll .prose {
  margin: 0 !important;
}

.artifact-section {
  border: 1px solid rgba(15, 23, 42, 0.07);
  border-radius: 12px;
  background: rgba(248, 250, 252, 0.88);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
}

.artifact-section > .wrap {
  padding: 0.45rem 0.5rem !important;
  gap: 0.28rem;
}

.preview-scroll {
  flex: 1 1 auto;
  max-height: calc(100vh - 190px);
  margin-top: 0.18rem;
}

.preview-audio-fixed {
  flex: 0 0 auto;
  margin-top: 0.18rem;
}

.preview-scroll > .wrap {
  min-height: 0;
  gap: 0.35rem;
}

.script-editor-panel {
  margin-top: 0.28rem;
  gap: 0.3rem;
}

.script-insert-row {
  margin: 0.15rem 0;
}

.script-insert-button button {
  border-style: dashed !important;
  border-radius: 12px !important;
}

.script-turn-row {
  gap: 0.32rem;
  padding: 0.38rem 0.05rem 0.45rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}

.script-turn-row:last-child {
  border-bottom: none;
}

.script-turn-header {
  align-items: flex-end;
  gap: 0.38rem;
}

.script-turn-speaker {
  max-width: 14rem;
}

.script-turn-content {
  width: 100%;
}

.script-turn-content textarea {
  border-radius: 12px !important;
}

.script-turn-delete button {
  min-width: 4.4rem;
  border-radius: 12px !important;
}

.script-turn-empty {
  color: #64748b;
  padding: 0.35rem 0 0.1rem;
}
"""

AUDIO_PREVIEW_HTML_TEMPLATE = """
<div class="anypod-audio-preview" data-audio-path="${value || ''}">
  <div class="anypod-audio-toolbar">
    <button type="button" class="anypod-audio-play">Play</button>
    <div class="anypod-audio-times">
      <span class="anypod-audio-current">00:00</span>
      <span class="anypod-audio-divider">/</span>
      <span class="anypod-audio-duration">--:--</span>
    </div>
  </div>
  <div class="anypod-audio-waveform"></div>
  <div class="anypod-audio-status">Preparing audio preview...</div>
  <audio class="anypod-audio-fallback" controls hidden preload="metadata"></audio>
</div>
"""

AUDIO_PREVIEW_CSS_TEMPLATE = """
.anypod-audio-preview {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  padding: 0.15rem 0 0.35rem;
}

.anypod-audio-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.anypod-audio-play {
  border: 1px solid rgba(8, 145, 178, 0.22);
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(236, 254, 255, 0.95), rgba(207, 250, 254, 0.88));
  color: #0f172a;
  font-weight: 600;
  min-width: 4.8rem;
  padding: 0.45rem 0.95rem;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.anypod-audio-play:hover {
  transform: translateY(-1px);
  border-color: rgba(8, 145, 178, 0.38);
  box-shadow: 0 10px 24px rgba(8, 145, 178, 0.14);
}

.anypod-audio-times {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  color: #475569;
  font-variant-numeric: tabular-nums;
  font-size: 0.96rem;
}

.anypod-audio-waveform {
  min-height: 112px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 16px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 249, 0.96)),
    radial-gradient(circle at top left, rgba(8, 145, 178, 0.10), transparent 34%);
  padding: 0.8rem 0.55rem;
  overflow: hidden;
}

.anypod-audio-status {
  min-height: 1.25rem;
  color: #64748b;
  font-size: 0.92rem;
}

.anypod-audio-fallback {
  width: 100%;
  margin-top: 0.1rem;
}
"""

AUDIO_PREVIEW_JS_ON_LOAD = """
const scriptId = "anypod-wavesurfer-script";
const timerKey = "__anypodAudioPreviewTimer";
const waveKey = "__anypodAudioPreviewWave";
const urlKey = "__anypodAudioPreviewUrl";
const renderKey = "__anypodAudioPreviewRenderId";
const loadingKey = "__anypodAudioPreviewLoading";
const fallbackKey = "__anypodAudioPreviewFallbackPath";

const uiTexts = {
  en: {
    play: "Play",
    pause: "Pause",
    empty: "No audio is available for preview.",
    loading: "Loading the full waveform...",
    ready: "Full waveform loaded. Click the waveform to seek.",
    waveFail: "Waveform preview failed to load. Switched to the native audio player.",
    libFail: "Waveform library failed to load. Switched to the native audio player.",
  },
  zh: {
    play: "播放",
    pause: "暂停",
    empty: "暂无音频可预览。",
    loading: "正在加载完整波形...",
    ready: "完整波形已加载，可点击波形跳转进度。",
    waveFail: "波形加载失败，已切换为原生音频播放器。",
    libFail: "波形库加载失败，已切换为原生音频播放器。",
  },
};

const getUiLang = () => (document.documentElement.dataset.anypodUiLang === "zh" ? "zh" : "en");
const t = (key) => (uiTexts[getUiLang()] || uiTexts.en)[key] || uiTexts.en[key] || key;

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "--:--";
  }
  const rounded = Math.floor(seconds);
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
};

const destroyWave = () => {
  const existing = element[waveKey];
  if (existing) {
    try {
      existing.destroy();
    } catch (error) {
      console.warn("Destroy WaveSurfer failed:", error);
    }
    element[waveKey] = null;
  }
};

const ensureWaveSurfer = () => new Promise((resolve, reject) => {
  if (window.WaveSurfer && typeof window.WaveSurfer.create === "function") {
    resolve(window.WaveSurfer);
    return;
  }

  const existingScript = document.getElementById(scriptId);
  if (existingScript) {
    existingScript.addEventListener("load", () => resolve(window.WaveSurfer), { once: true });
    existingScript.addEventListener("error", () => reject(new Error("WaveSurfer script load failed")), { once: true });
    return;
  }

  const script = document.createElement("script");
  script.id = scriptId;
  script.src = "https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js";
  script.async = true;
  script.onload = () => resolve(window.WaveSurfer);
  script.onerror = () => reject(new Error("WaveSurfer script load failed"));
  document.head.appendChild(script);
});

const renderPreview = async () => {
  const root = element.querySelector(".anypod-audio-preview");
  if (!root) {
    destroyWave();
    element[loadingKey] = "";
    element[fallbackKey] = "";
    return;
  }

  const renderId = (element[renderKey] || 0) + 1;
  element[renderKey] = renderId;

  const encodedPath = root.dataset.audioPath || "";
  const waveform = root.querySelector(".anypod-audio-waveform");
  const playButton = root.querySelector(".anypod-audio-play");
  const currentTime = root.querySelector(".anypod-audio-current");
  const duration = root.querySelector(".anypod-audio-duration");
  const status = root.querySelector(".anypod-audio-status");
  const fallback = root.querySelector(".anypod-audio-fallback");

  if (!waveform || !playButton || !currentTime || !duration || !status || !fallback) {
    destroyWave();
    return;
  }

  if (!encodedPath) {
    destroyWave();
    waveform.innerHTML = "";
    fallback.hidden = true;
    fallback.removeAttribute("src");
    currentTime.textContent = "00:00";
    duration.textContent = "--:--";
    playButton.textContent = t("play");
    status.dataset.state = "empty";
    status.textContent = t("empty");
    element[urlKey] = "";
    element[loadingKey] = "";
    element[fallbackKey] = "";
    return;
  }

  if (element[urlKey] === encodedPath && element[waveKey]) {
    return;
  }

  if (element[loadingKey] === encodedPath) {
    return;
  }

  element[loadingKey] = encodedPath;
  element[urlKey] = encodedPath;
  destroyWave();
  waveform.innerHTML = "";
  fallback.hidden = true;
  fallback.removeAttribute("src");
  currentTime.textContent = "00:00";
  duration.textContent = "--:--";
  playButton.textContent = t("play");
  status.dataset.state = "loading";
  status.textContent = t("loading");

  const baseUrl = new URL("./", window.location.href);
  const audioUrl = new URL(`gradio_api/file=${encodedPath}`, baseUrl).toString();

  playButton.onclick = null;

  try {
    const WaveSurfer = await ensureWaveSurfer();
    if (!WaveSurfer || typeof WaveSurfer.create !== "function") {
      throw new Error("WaveSurfer unavailable");
    }

    if (element[renderKey] !== renderId) {
      return;
    }

    const wave = WaveSurfer.create({
      container: waveform,
      url: audioUrl,
      height: 84,
      waveColor: "#94a3b8",
      progressColor: "#0891b2",
      cursorColor: "#0f172a",
      cursorWidth: 2,
      barWidth: 2,
      barGap: 1.5,
      barRadius: 2,
      normalize: true,
      fillParent: true,
      minPxPerSec: 0,
      autoCenter: false,
      autoScroll: false,
      dragToSeek: true,
      hideScrollbar: true,
      interact: true,
    });

    if (element[renderKey] !== renderId) {
      wave.destroy();
      return;
    }

    element[waveKey] = wave;
    element[loadingKey] = "";
    element[fallbackKey] = "";

    playButton.onclick = () => {
      wave.playPause();
    };

    wave.on("decode", (seconds) => {
      duration.textContent = formatTime(seconds);
    });

    wave.on("ready", () => {
      duration.textContent = formatTime(wave.getDuration());
      currentTime.textContent = "00:00";
      playButton.textContent = t("play");
      status.dataset.state = "ready";
      status.textContent = t("ready");
    });

    wave.on("timeupdate", (seconds) => {
      currentTime.textContent = formatTime(seconds);
    });

    wave.on("play", () => {
      playButton.textContent = t("pause");
    });

    wave.on("pause", () => {
      playButton.textContent = t("play");
    });

    wave.on("finish", () => {
      playButton.textContent = t("play");
      currentTime.textContent = formatTime(wave.getDuration());
    });

    wave.on("error", () => {
      destroyWave();
      fallback.hidden = false;
      fallback.src = audioUrl;
      playButton.textContent = t("play");
      status.dataset.state = "waveFail";
      status.textContent = t("waveFail");
      element[loadingKey] = "";
      element[fallbackKey] = encodedPath;
    });
  } catch (error) {
    destroyWave();
    fallback.hidden = false;
    fallback.src = audioUrl;
    playButton.textContent = t("play");
    status.dataset.state = "libFail";
    status.textContent = t("libFail");
    element[loadingKey] = "";
    element[fallbackKey] = encodedPath;
    console.warn("Load WaveSurfer failed:", error);
  }
};

const syncPreview = () => {
  const root = element.querySelector(".anypod-audio-preview");
  if (!root) {
    if (element[urlKey]) {
      element[urlKey] = "";
      destroyWave();
    }
    element[loadingKey] = "";
    element[fallbackKey] = "";
    return;
  }

  const encodedPath = root.dataset.audioPath || "";
  const playButton = root.querySelector(".anypod-audio-play");
  const status = root.querySelector(".anypod-audio-status");
  if (playButton && status) {
    const state = status.dataset.state || "";
    if (state === "empty") {
      playButton.textContent = t("play");
      status.textContent = t("empty");
    } else if (state === "loading") {
      playButton.textContent = t("play");
      status.textContent = t("loading");
    } else if (state === "ready") {
      const wave = element[waveKey];
      playButton.textContent = wave && typeof wave.isPlaying === "function" && wave.isPlaying() ? t("pause") : t("play");
      status.textContent = t("ready");
    } else if (state === "waveFail") {
      playButton.textContent = t("play");
      status.textContent = t("waveFail");
    } else if (state === "libFail") {
      playButton.textContent = t("play");
      status.textContent = t("libFail");
    }
  }
  if (encodedPath !== (element[urlKey] || "")) {
    queueMicrotask(renderPreview);
    return;
  }

  if (
    encodedPath &&
    !element[waveKey] &&
    element[loadingKey] !== encodedPath &&
    element[fallbackKey] !== encodedPath
  ) {
    queueMicrotask(renderPreview);
  }
};

if (!element[timerKey]) {
  element[timerKey] = window.setInterval(syncPreview, 400);
}

queueMicrotask(renderPreview);
"""


def _clean_text(value: str | None) -> str:
    return str(value or "").strip()


def _optional_text(value: str | None) -> str | None:
    text = _clean_text(value)
    return text or None


def _normalize_ui_language(value: str | None, default: str = "en") -> str:
    text = _clean_text(value).lower()
    if text in UI_LANGUAGES:
        return text
    return default


def _ui_text(lang: str | None, key: str, **kwargs: object) -> str:
    normalized_lang = _normalize_ui_language(lang)
    template = UI_TEXTS.get(normalized_lang, UI_TEXTS["en"]).get(key, UI_TEXTS["en"].get(key, key))
    return template.format(**kwargs)


def _choice_pairs(choice_map: dict[str, dict[str, str]], lang: str | None) -> list[tuple[str, str]]:
    normalized_lang = _normalize_ui_language(lang)
    labels = choice_map.get(normalized_lang, choice_map["en"])
    return [(label, value) for value, label in labels.items()]


_VOICE_UPDATE_UNSET = object()


def _voice_choice_pairs() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for entry in load_voice_library():
        voice_id = _clean_text(entry.get("voice_id"))
        if not voice_id:
            continue
        speaker_name = _clean_text(entry.get("speaker_name"))
        label = f"{voice_id} | {speaker_name}" if speaker_name else voice_id
        choices.append((label, voice_id))
    return choices


def _resolve_voice_dropdown_value(value: str | None) -> str | None:
    normalized_value = _clean_text(value)
    if not normalized_value:
        return None
    valid_values = {choice_value for _, choice_value in _voice_choice_pairs()}
    if normalized_value in valid_values:
        return normalized_value
    return None


def _build_voice_dropdown_update(label: str | None = None, value: object = _VOICE_UPDATE_UNSET) -> dict:
    update_kwargs: dict[str, object] = {
        "choices": _voice_choice_pairs(),
    }
    if label is not None:
        update_kwargs["label"] = label
    if value is not _VOICE_UPDATE_UNSET:
        normalized_value = value if isinstance(value, str) or value is None else None
        update_kwargs["value"] = _resolve_voice_dropdown_value(normalized_value)
    return gr.update(**update_kwargs)


def _normalize_language_value(value: str | None, default: str) -> str:
    text = _clean_text(value)
    if not text:
        return default
    if text in LANGUAGE_LABEL_TO_CODE:
        return LANGUAGE_LABEL_TO_CODE[text]
    if text in LANGUAGE_CODE_TO_LABEL:
        return text
    return default


def _language_dropdown_value(default_code: str) -> str:
    return default_code if default_code in CONTENT_LANGUAGE_UI_LABELS["en"] else "zh"


def _normalize_dialogue_mode_value(value: str | None, default: str) -> str:
    text = _clean_text(value)
    if not text:
        return default
    if text in DIALOGUE_MODE_LABEL_TO_CODE:
        return DIALOGUE_MODE_LABEL_TO_CODE[text]
    if text in DIALOGUE_MODE_CODE_TO_LABEL:
        return text
    return default


def _dialogue_mode_dropdown_value(default_code: str) -> str:
    return default_code if default_code in DIALOGUE_MODE_UI_LABELS["en"] else "dual"


def _normalize_program_mode_value(value: str | None, default: str) -> str:
    text = _clean_text(value)
    if not text:
        return default
    if text in PROGRAM_MODE_LABEL_TO_CODE:
        return PROGRAM_MODE_LABEL_TO_CODE[text]
    if text in PROGRAM_MODE_CODE_TO_LABEL:
        return text
    return default


def _program_mode_dropdown_value(default_code: str) -> str:
    return default_code if default_code in PROGRAM_MODE_UI_LABELS["en"] else "faithful"


def _to_int(value: float | int | None, default: int) -> int:
    return int(default if value is None else value)


def _to_optional_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _to_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_input_path(uploaded_file: str | None, input_path_text: str) -> Path:
    if uploaded_file:
        return Path(uploaded_file).expanduser().resolve()

    input_text = _clean_text(input_path_text)
    if not input_text:
        raise ValueError("Provide an absolute input path or select a PDF/TXT file with the upload component.")
    return Path(input_text).expanduser().resolve()


def _resolve_output_dir(output_dir_text: str, input_path: Path) -> Path:
    requested_output_dir = _clean_text(output_dir_text)
    if requested_output_dir:
        return Path(requested_output_dir).expanduser().resolve()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    auto_dir_name = f"{build_book_id(input_path)}_{timestamp}"
    return (DEFAULT_OUTPUT_ROOT / auto_dir_name).resolve()


def build_runtime_args(
    uploaded_file: str | None,
    input_path_text: str,
    output_dir_text: str,
    dialogue_mode: str,
    primary_language: str,
    mode: str,
    target_episode_level: str,
    tts_backend: str,
    allow_external_knowledge: bool,
    show_title: str,
    positioning: str,
    target_audience: str,
    fixed_opening: str,
    speaker_1_name: str,
    speaker_1_style: str,
    speaker_1_voice_id: str,
    speaker_1_language: str,
    speaker_2_name: str,
    speaker_2_style: str,
    speaker_2_voice_id: str,
    speaker_2_language: str,
    chunk_max_words: float | int | None,
    step2_num_workers: float | int | None,
    max_retries: float | int | None,
) -> argparse.Namespace:
    input_path = _resolve_input_path(uploaded_file, input_path_text)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    args = build_parser().parse_args([])
    args.input_path = input_path
    args.output_dir = _resolve_output_dir(output_dir_text, input_path)
    args.chunk_max_words = _to_int(chunk_max_words, DEFAULT_ARGS.chunk_max_words)
    args.ignore_bookmarks = bool(DEFAULT_ARGS.ignore_bookmarks)
    args.force = bool(DEFAULT_ARGS.force)
    args.dialogue_mode = _normalize_dialogue_mode_value(dialogue_mode, DEFAULT_ARGS.dialogue_mode)
    args.primary_language = _normalize_language_value(primary_language, DEFAULT_ARGS.primary_language)
    args.mode = _normalize_program_mode_value(mode, DEFAULT_ARGS.mode)
    args.target_episode_level = target_episode_level or DEFAULT_ARGS.target_episode_level
    args.allow_external_knowledge = bool(allow_external_knowledge)
    args.show_title = _clean_text(show_title)
    args.positioning = _clean_text(positioning)
    args.target_audience = _clean_text(target_audience)
    args.fixed_opening = _clean_text(fixed_opening)
    args.speaker_1_name = _clean_text(speaker_1_name)
    args.speaker_1_style = _clean_text(speaker_1_style)
    args.speaker_1_voice_id = _clean_text(speaker_1_voice_id) or DEFAULT_ARGS.speaker_1_voice_id
    args.speaker_1_language = _normalize_language_value(speaker_1_language, args.primary_language)
    args.speaker_2_name = _clean_text(speaker_2_name)
    args.speaker_2_style = _clean_text(speaker_2_style)
    args.speaker_2_voice_id = _clean_text(speaker_2_voice_id) or DEFAULT_ARGS.speaker_2_voice_id
    args.speaker_2_language = _normalize_language_value(speaker_2_language, args.primary_language)
    args.episode_ids = []
    args.skip_tts = bool(DEFAULT_ARGS.skip_tts)
    args.tts_backend = tts_backend or DEFAULT_ARGS.tts_backend
    if args.tts_backend in {"moss-tts", "moss-tts(api)"} and args.dialogue_mode == "dual":
        raise ValueError(f"{args.tts_backend} 后端仅支持单人模式 (dialogue_mode=single)，不支持双人模式")
    args.max_retries = _to_int(max_retries, GRADIO_DEFAULT_MAX_RETRIES)
    args.step2_num_workers = _to_int(step2_num_workers, GRADIO_DEFAULT_STEP2_NUM_WORKERS)
    args.log_level = "WARNING"
    return args


def _list_output_files(directory: Path, pattern: str) -> list[str]:
    if not directory.exists():
        return []
    return [str(path.resolve()) for path in sorted(directory.glob(pattern)) if path.is_file()]


def _collect_artifacts(output_dir: Path) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    metadata_files = [
        output_dir / "book_structure.json",
        output_dir / "book_charter.json",
        output_dir / "program_config.json",
        output_dir / "speaker_bible.json",
        output_dir / "episode_plan.json",
        output_dir / "series_memory_summary.json",
    ]
    artifact_files = [str(path.resolve()) for path in metadata_files if path.is_file()]
    script_files = _list_output_files(output_dir / "scripts", "*.txt")
    audio_files = _list_output_files(output_dir / "audios", "*.wav")
    return artifact_files or None, script_files or None, audio_files or None


def _prepare_files_for_gradio(
    artifact_files: list[str] | None,
    script_files: list[str] | None,
    audio_files: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    return artifact_files, script_files, audio_files


def _register_static_output_dir(output_dir: Path) -> None:
    gr.set_static_paths([output_dir.resolve()])


def _collect_live_outputs(
    output_dir: Path | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    if output_dir is None or not output_dir.exists():
        return None, None, None
    _register_static_output_dir(output_dir)
    artifact_files, script_files, audio_files = _collect_artifacts(output_dir)
    return _prepare_files_for_gradio(artifact_files, script_files, audio_files)


def _read_text_preview(path_value: str | None, empty_message: str) -> str:
    if not path_value:
        return empty_message
    path = Path(path_value)
    if not path.is_file():
        return f"File does not exist: {path}"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Read failed: {exc}"


def _speaker_label(name: str | None, fallback: str) -> str:
    return _clean_text(name) or fallback


def _resolve_output_dir_for_preview(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.parent.name in {"scripts", "audios"}:
        return path.parent.parent
    return None


def _read_speaker_display_names(path_value: str | None) -> tuple[str | None, str | None]:
    output_dir = _resolve_output_dir_for_preview(path_value)
    if output_dir is None:
        return None, None
    speaker_bible_path = output_dir / "speaker_bible.json"
    if not speaker_bible_path.is_file():
        return None, None
    try:
        speaker_bible = json.loads(speaker_bible_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    speakers = speaker_bible.get("speakers")
    if not isinstance(speakers, list):
        return None, None
    speaker_1_name = None
    speaker_2_name = None
    if len(speakers) >= 1 and isinstance(speakers[0], dict):
        speaker_1_name = _optional_text(speakers[0].get("display_name"))
    if len(speakers) >= 2 and isinstance(speakers[1], dict):
        speaker_2_name = _optional_text(speakers[1].get("display_name"))
    return speaker_1_name, speaker_2_name


def _build_speaker_label_map(path_value: str | None, ui_lang: str = "en") -> dict[str, str]:
    speaker_1_name, speaker_2_name = _read_speaker_display_names(path_value)
    return {
        "[S1]": _speaker_label(speaker_1_name, _ui_text(ui_lang, "host_1_fallback")),
        "[S2]": _speaker_label(speaker_2_name, _ui_text(ui_lang, "host_2_fallback")),
    }


def _build_speaker_tag_map(path_value: str | None, ui_lang: str = "en") -> dict[str, str]:
    label_map = _build_speaker_label_map(path_value, ui_lang)
    tag_map = {label: tag for tag, label in label_map.items()}
    tag_map.setdefault(_ui_text(ui_lang, "host_1_fallback"), "[S1]")
    tag_map.setdefault(_ui_text(ui_lang, "host_2_fallback"), "[S2]")
    return tag_map


def _build_speaker_choices(path_value: str | None, ui_lang: str = "en") -> list[str]:
    label_map = _build_speaker_label_map(path_value, ui_lang)
    choices: list[str] = []
    for tag in ("[S1]", "[S2]"):
        label = _clean_text(label_map.get(tag))
        if label and label not in choices:
            choices.append(label)
    if not choices:
        choices.append(_ui_text(ui_lang, "host_1_fallback"))
    return choices


def _parse_script_turns(script_text: str) -> list[dict[str, str]]:
    normalized_text = str(script_text or "").strip()
    if not normalized_text:
        return []

    matches = list(SCRIPT_TURN_TAG_PATTERN.finditer(normalized_text))
    if not matches:
        return [{"tag": "[S1]", "content": normalized_text}]

    turns: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        tag = match.group(0)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start:end].strip()
        if content:
            turns.append({"tag": tag, "content": content})
    return turns


def _normalize_editor_rows(value: object) -> list[list[str]]:
    if value is None:
        return []
    if hasattr(value, "values"):
        try:
            value = value.values.tolist()
        except Exception:
            pass
    if not isinstance(value, list):
        return []

    rows: list[list[str]] = []
    for row in value:
        if isinstance(row, (list, tuple)):
            speaker = _clean_text(row[0] if len(row) >= 1 else "")
            content = _clean_text(row[1] if len(row) >= 2 else "")
            rows.append([speaker, content])
    return rows


def _build_script_editor_rows(
    path_value: str | None,
    ui_lang: str = "en",
) -> tuple[list[list[str]], list[str], str | None]:
    speaker_choices = _build_speaker_choices(path_value, ui_lang)
    if not path_value:
        return [], speaker_choices, _ui_text(ui_lang, "editable_script_missing")
    path = Path(path_value)
    if not path.is_file():
        return [], speaker_choices, _ui_text(ui_lang, "script_file_missing", path=path)

    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], speaker_choices, _ui_text(ui_lang, "script_read_failed", error=exc)

    label_map = _build_speaker_label_map(path_value, ui_lang)
    turns = _parse_script_turns(raw_text)
    rows = [[label_map.get(turn["tag"], turn["tag"]), turn["content"]] for turn in turns if turn["content"]]
    if not rows:
        return [], speaker_choices, _ui_text(ui_lang, "script_empty")
    return rows, speaker_choices, None


def _normalize_script_editor_state(
    rows_value: object,
    speaker_choices_value: object,
) -> tuple[list[list[str]], list[str]]:
    normalized_choices = []
    if isinstance(speaker_choices_value, list):
        for item in speaker_choices_value:
            text = _clean_text(str(item))
            if text and text not in normalized_choices:
                normalized_choices.append(text)
    if not normalized_choices:
        normalized_choices = [_ui_text("en", "host_1_fallback")]

    normalized_rows: list[list[str]] = []
    for speaker_label, content in _normalize_editor_rows(rows_value):
        normalized_speaker = speaker_label if speaker_label in normalized_choices else normalized_choices[0]
        normalized_rows.append([normalized_speaker, content])
    return normalized_rows, normalized_choices


def _set_script_editor_row(
    rows_value: object,
    speaker_choices_value: object,
    row_index: int,
    speaker_label: str | None,
    content: str | None,
) -> list[list[str]]:
    rows, speaker_choices = _normalize_script_editor_state(rows_value, speaker_choices_value)
    if row_index < 0 or row_index >= len(rows):
        return rows
    normalized_speaker = _clean_text(speaker_label)
    if normalized_speaker not in speaker_choices:
        normalized_speaker = rows[row_index][0] if rows[row_index][0] in speaker_choices else speaker_choices[0]
    rows[row_index] = [normalized_speaker, _clean_text(content)]
    return rows


def _delete_script_editor_row(
    rows_value: object,
    row_index: int,
) -> list[list[str]]:
    rows = _normalize_editor_rows(rows_value)
    if 0 <= row_index < len(rows):
        del rows[row_index]
    return rows


def _append_script_editor_row(
    rows_value: object,
    speaker_choices_value: object,
) -> list[list[str]]:
    rows, speaker_choices = _normalize_script_editor_state(rows_value, speaker_choices_value)
    default_speaker = speaker_choices[0] if speaker_choices else _ui_text("en", "host_1_fallback")
    rows.append([default_speaker, ""])
    return rows


def _collect_script_rows_from_component_values(
    row_count: int | float | None,
    speaker_choices_value: object,
    component_values: tuple[object, ...],
) -> tuple[list[list[str]], list[str]]:
    normalized_row_count = max(int(row_count or 0), 0)
    rows, speaker_choices = _normalize_script_editor_state([], speaker_choices_value)
    speaker_values = component_values[:normalized_row_count]
    content_values = component_values[normalized_row_count : normalized_row_count * 2]
    rebuilt_rows: list[list[str]] = []
    default_speaker = speaker_choices[0] if speaker_choices else _ui_text("en", "host_1_fallback")
    for index in range(normalized_row_count):
        speaker_label = _clean_text(speaker_values[index] if index < len(speaker_values) else default_speaker)
        if speaker_label not in speaker_choices:
            speaker_label = default_speaker
        content = _clean_text(content_values[index] if index < len(content_values) else "")
        rebuilt_rows.append([speaker_label, content])
    return rebuilt_rows, speaker_choices


def _insert_script_editor_row_from_components(
    row_count: int | float | None,
    speaker_choices_value: object,
    insert_index: int,
    *component_values: object,
) -> list[list[str]]:
    rows, speaker_choices = _collect_script_rows_from_component_values(
        row_count,
        speaker_choices_value,
        component_values,
    )
    normalized_insert_index = max(0, min(int(insert_index), len(rows)))
    default_speaker = speaker_choices[0] if speaker_choices else _ui_text("en", "host_1_fallback")
    rows.insert(normalized_insert_index, [default_speaker, ""])
    return rows


def _delete_script_editor_row_from_components(
    row_count: int | float | None,
    speaker_choices_value: object,
    delete_index: int,
    *component_values: object,
) -> list[list[str]]:
    rows, _ = _collect_script_rows_from_component_values(
        row_count,
        speaker_choices_value,
        component_values,
    )
    if 0 <= int(delete_index) < len(rows):
        del rows[int(delete_index)]
    return rows


def _save_script_text_from_components(
    script_path_value: str | None,
    speaker_choices_value: object,
    ui_lang: str,
    row_count: int | float | None,
    *component_values: object,
) -> tuple[dict, dict]:
    rows, _ = _collect_script_rows_from_component_values(
        row_count,
        speaker_choices_value,
        component_values,
    )
    return _save_script_text(script_path_value, rows, ui_lang)


def _serialize_script_editor_rows(rows: list[list[str]], path_value: str | None, ui_lang: str = "en") -> str:
    tag_map = _build_speaker_tag_map(path_value, ui_lang)
    serialized_parts: list[str] = []
    for speaker_label, content in rows:
        normalized_speaker_label = _clean_text(speaker_label)
        normalized_content = _clean_text(content)
        if not normalized_content:
            continue
        tag = tag_map.get(normalized_speaker_label)
        if not tag:
            continue
        serialized_parts.append(f"{tag}{normalized_content}")
    return "".join(serialized_parts)


def _resolve_tts_backend_for_episode(
    output_dir: Path,
    episode_id: str,
    fallback: str,
) -> str:
    candidate_paths = [output_dir / "audio_meta" / f"{episode_id}.json"]
    audio_meta_dir = output_dir / "audio_meta"
    if audio_meta_dir.exists():
        candidate_paths.extend(sorted(audio_meta_dir.glob("*.json")))
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        backend = _optional_text(payload.get("tts_backend")) if isinstance(payload, dict) else None
        if backend in {"moss-ttsd", "vibevoice", "moss-tts"}:
            return backend
    return fallback


def _read_audio_preview(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    return str(path.resolve())


def _build_audio_preview_value(path_value: str | None) -> str | None:
    audio_path = _read_audio_preview(path_value)
    if audio_path is None:
        return None
    return urllib.parse.quote(audio_path, safe="/:")


def _resolve_script_for_audio(audio_path_value: str | None) -> str | None:
    if not audio_path_value:
        return None
    audio_path = Path(audio_path_value)
    if audio_path.parent.name != "audios":
        return None
    script_path = audio_path.parent.parent / "scripts" / f"{audio_path.stem}.txt"
    if not script_path.is_file():
        return None
    return str(script_path.resolve())


def _extract_path_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, str):
        return _optional_text(value)
    if isinstance(value, dict):
        return _optional_text(value.get("path")) or _optional_text(value.get("url"))
    path_attr = getattr(value, "path", None)
    if path_attr is not None:
        return _optional_text(str(path_attr))
    return None


def _extract_existing_path(value: object) -> str | None:
    path_value = _extract_path_value(value)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    return str(path.resolve())


def _resolve_selected_file(
    component_value: str | list[str] | None,
    evt: gr.SelectData | None,
) -> str | None:
    if isinstance(component_value, list) and evt is not None and isinstance(getattr(evt, "index", None), int):
        index = evt.index
        if 0 <= index < len(component_value):
            selected_path = _extract_existing_path(component_value[index])
            if selected_path:
                return selected_path
    selected_from_value = _extract_existing_path(component_value)
    if selected_from_value:
        return selected_from_value
    if evt is not None:
        selected_path = _extract_existing_path(getattr(evt, "value", None))
        if selected_path:
            return selected_path
    return _extract_path_value(component_value)


def _empty_preview_response(
    ui_lang: str = "en",
) -> tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None]:
    return (
        gr.update(visible=False),
        _ui_text(ui_lang, "preview_default_title"),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=""),
        gr.update(visible=False),
        None,
        None,
        gr.update(visible=False, value=""),
        None,
    )


def _build_script_preview_response(
    preview_title: str,
    script_path_value: str | None,
    ui_lang: str = "en",
    download_path_value: str | None = None,
    audio_path_value: str | None = None,
) -> tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None]:
    if not script_path_value and not audio_path_value:
        return _empty_preview_response(ui_lang)

    script_rows, speaker_choices, script_message = _build_script_editor_rows(script_path_value, ui_lang)
    audio_preview_value = _build_audio_preview_value(audio_path_value)
    visible_script = script_path_value is not None
    visible_audio = audio_preview_value is not None
    resolved_download_path = download_path_value or script_path_value or audio_path_value
    title_path = resolved_download_path or script_path_value or audio_path_value
    title_suffix = f"\n\n`{Path(title_path).name}`" if title_path else ""

    return (
        gr.update(visible=True),
        f"### {preview_title}{title_suffix}",
        gr.update(visible=bool(resolved_download_path), value=resolved_download_path),
        gr.update(visible=visible_audio, value=audio_preview_value or ""),
        gr.update(visible=visible_script),
        script_rows,
        speaker_choices,
        gr.update(visible=bool(script_message), value=script_message or ""),
        script_path_value if visible_script else None,
    )


def _build_text_preview_response(
    preview_title: str,
    path_value: str | None,
    ui_lang: str,
    empty_message: str,
) -> tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None]:
    if not path_value:
        return (
            gr.update(visible=False),
            _ui_text(ui_lang, "preview_default_title"),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=""),
            gr.update(visible=False),
            None,
            None,
            gr.update(visible=True, value=empty_message),
            None,
        )
    return _build_script_preview_response(
        preview_title=preview_title,
        script_path_value=path_value,
        ui_lang=ui_lang,
        download_path_value=path_value,
        audio_path_value=None,
    )


def _build_audio_preview_response(
    path_value: str | None,
    ui_lang: str,
) -> tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None]:
    if not path_value:
        return _empty_preview_response(ui_lang)
    audio_path = _read_audio_preview(path_value)
    if audio_path is None:
        return _empty_preview_response(ui_lang)
    script_path = _resolve_script_for_audio(audio_path)
    return _build_script_preview_response(
        preview_title=_ui_text(ui_lang, "audio_preview_title"),
        script_path_value=script_path,
        ui_lang=ui_lang,
        download_path_value=audio_path,
        audio_path_value=audio_path,
    )


def _format_status(title: str, output_dir: Path | None, detail: str, ui_lang: str = "en") -> str:
    parts = [f"### {title}", detail]
    if output_dir is not None:
        parts.append(f"{_ui_text(ui_lang, 'status_output_dir')}: `{output_dir}`")
    return "\n\n".join(parts)


class _LogBuffer:
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._version = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._chunks.append(text)
            self._version += 1
        return len(text)

    def flush(self) -> None:
        return None

    def snapshot(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def snapshot_and_version(self) -> tuple[str, int]:
        with self._lock:
            return "".join(self._chunks), self._version


class _TeeWriter:
    def __init__(self, buffer: _LogBuffer, mirror_stream) -> None:
        self.buffer = buffer
        self.mirror_stream = mirror_stream

    def write(self, text: str) -> int:
        written = self.buffer.write(text)
        if self.mirror_stream is not None and text:
            self.mirror_stream.write(text)
            self.mirror_stream.flush()
        return written

    def flush(self) -> None:
        if self.mirror_stream is not None:
            self.mirror_stream.flush()


class _LocalRegenWorker:
    def __init__(
        self,
        state: "_PipelineState",
        output_dir: Path,
        tts_backend: str,
    ) -> None:
        self.state = state
        self.output_dir = output_dir.resolve()
        self.tts_backend = tts_backend
        self._task_queue: Queue[str] = Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending_episode_ids: list[str] = []
        self._current_episode_id: str | None = None
        self._active = False

    def enqueue(self, episode_id: str) -> str:
        normalized_episode_id = _clean_text(episode_id)
        if not normalized_episode_id:
            return "Missing episode ID for regeneration."

        start_thread = False
        with self._lock:
            if normalized_episode_id == self._current_episode_id or normalized_episode_id in self._pending_episode_ids:
                return f"{normalized_episode_id} is already queued for local regeneration."
            self._pending_episode_ids.append(normalized_episode_id)
            self._task_queue.put(normalized_episode_id)
            if self._thread is None or not self._thread.is_alive():
                self._active = True
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name=f"anypod_local_regen_{self.output_dir.name}",
                    daemon=True,
                )
                start_thread = True

        if start_thread:
            self._thread.start()
        return f"{normalized_episode_id} was added to the local regeneration queue."

    def is_active(self) -> bool:
        with self._lock:
            return self._active or bool(self._pending_episode_ids) or self._current_episode_id is not None

    def snapshot(self) -> tuple[str | None, list[str], str]:
        with self._lock:
            return self._current_episode_id, list(self._pending_episode_ids), self.tts_backend

    def _run_loop(self) -> None:
        step7_tts = Step7TTS(
            output_dir=self.output_dir,
            tts_backend=self.tts_backend,
        )
        try:
            speaker_bible = step7_tts.prepare_runtime()
            while True:
                try:
                    episode_id = self._task_queue.get(timeout=0.3)
                except Empty:
                    with self._lock:
                        if not self._pending_episode_ids and self._current_episode_id is None:
                            break
                    continue

                with self._lock:
                    if episode_id in self._pending_episode_ids:
                        self._pending_episode_ids.remove(episode_id)
                    self._current_episode_id = episode_id

                t0 = time.time()
                _emit_state_text(self.state, f"[AnyPod UI] Starting local audio regeneration: {episode_id}\n")
                try:
                    step7_tts.synthesize_episode(
                        episode_id=episode_id,
                        speaker_bible=speaker_bible,
                    )
                    _emit_state_text(
                        self.state,
                        f"[AnyPod UI] Finished local audio regeneration: {episode_id} ({time.time() - t0:.1f}s)\n",
                    )
                except Exception as exc:
                    _emit_state_text(self.state, f"[AnyPod UI] Local regeneration failed: {episode_id}, error: {exc}\n")
                finally:
                    with self._lock:
                        if self._current_episode_id == episode_id:
                            self._current_episode_id = None
                    self._task_queue.task_done()
        finally:
            step7_tts.close_tts_runtime()
            with self._lock:
                self._active = False


@dataclass
class _PipelineState:
    logs: _LogBuffer = field(default_factory=_LogBuffer)
    error_message: str | None = None
    done: bool = False
    output_dir: Path | None = None
    artifact_files: list[str] | None = None
    script_files: list[str] | None = None
    audio_files: list[str] | None = None
    finalized: bool = False
    meta_lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen[str] | None = None
    reader_thread: threading.Thread | None = None
    config_path: Path | None = None
    return_code: int | None = None
    termination_requested: bool = False
    terminated: bool = False
    local_regen_worker: _LocalRegenWorker | None = None


_JOB_LOCK = threading.Lock()
_JOBS: dict[str, _PipelineState] = {}


def _register_job(job_id: str, state: _PipelineState) -> None:
    with _JOB_LOCK:
        _JOBS[job_id] = state


def _get_job(job_id: str | None) -> _PipelineState | None:
    if not job_id:
        return None
    with _JOB_LOCK:
        return _JOBS.get(job_id)


def _build_running_response(
    output_dir: Path | None,
    logs: str,
    script_files: object,
    audio_files: object,
    job_id: str | None,
    ui_lang: str = "en",
    detail: str | None = None,
    stop_enabled: bool = True,
) -> tuple[str, str, object, object, str | None, gr.Timer, dict, dict]:
    resolved_detail = detail or _ui_text(ui_lang, "run_running_detail")
    return (
        _format_status(_ui_text(ui_lang, "status_running_title").removeprefix("### ").strip(), output_dir, resolved_detail, ui_lang),
        logs,
        script_files,
        audio_files,
        job_id,
        gr.Timer(value=POLL_INTERVAL_SEC, active=True),
        gr.update(interactive=False),
        gr.update(interactive=stop_enabled),
    )


def _build_finished_response(
    title: str,
    detail: str,
    output_dir: Path | None,
    logs: str,
    script_files: object,
    audio_files: object,
    job_id: str | None,
    ui_lang: str = "en",
) -> tuple[str, str, object, object, str | None, gr.Timer, dict, dict]:
    return (
        _format_status(title, output_dir, detail, ui_lang),
        logs,
        script_files,
        audio_files,
        job_id,
        gr.Timer(value=POLL_INTERVAL_SEC, active=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def _maybe_skip_list_update(
    current_files: list[str] | None,
    latest_files: list[str] | None,
) -> object:
    return gr.skip() if current_files == latest_files else latest_files


def _namespace_to_config_dict(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _write_runtime_config(args: argparse.Namespace) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix="anypod_gradio_", suffix=".json")
    os.close(fd)
    config_path = Path(raw_path)
    config_path.write_text(
        json.dumps(_namespace_to_config_dict(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path


def _build_pipeline_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--config_json",
        str(config_path),
    ]


def _emit_state_text(state: _PipelineState, text: str) -> None:
    _TeeWriter(state.logs, sys.__stdout__).write(text)


def _cleanup_job_files(state: _PipelineState) -> None:
    if state.config_path is not None:
        state.config_path.unlink(missing_ok=True)
        state.config_path = None


def _stream_process_output(state: _PipelineState) -> None:
    process = state.process
    if process is None or process.stdout is None:
        return

    writer = _TeeWriter(state.logs, sys.__stdout__)
    try:
        for raw_line in process.stdout:
            writer.write(raw_line)
    finally:
        process.stdout.close()


def _refresh_process_state(state: _PipelineState) -> None:
    process = state.process
    if process is None or state.done:
        return

    return_code = process.poll()
    if return_code is None:
        return

    state.return_code = return_code
    if state.reader_thread is not None:
        state.reader_thread.join(timeout=0.2)
    _cleanup_job_files(state)

    if state.termination_requested:
        state.terminated = True
    elif return_code != 0 and state.error_message is None:
        state.error_message = f"Main process exited with code {return_code}"

    state.done = True


def _launch_pipeline_process(state: _PipelineState, args: argparse.Namespace) -> None:
    state.output_dir = args.output_dir
    clear_tts_priority_queue(args.output_dir)
    state.config_path = _write_runtime_config(args)
    command = _build_pipeline_command(state.config_path)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    state.process = process
    _emit_state_text(state, f"[AnyPod UI] Started job process PID={process.pid}\n\n")
    state.reader_thread = threading.Thread(
        target=_stream_process_output,
        args=(state,),
        name=f"anypod_gradio_log_reader_{process.pid}",
        daemon=True,
    )
    state.reader_thread.start()


def _finalize_job_outputs(state: _PipelineState) -> None:
    assert state.output_dir is not None
    with state.meta_lock:
        if state.finalized:
            return
        artifact_files, script_files, audio_files = _collect_live_outputs(state.output_dir)
        state.artifact_files = artifact_files
        state.script_files = script_files
        state.audio_files = audio_files
        state.finalized = True


def _ensure_local_regen_worker(
    state: _PipelineState,
    output_dir: Path,
    tts_backend: str,
) -> _LocalRegenWorker:
    worker = state.local_regen_worker
    if (
        worker is None
        or worker.output_dir != output_dir.resolve()
        or (not worker.is_active() and worker.tts_backend != tts_backend)
    ):
        worker = _LocalRegenWorker(
            state=state,
            output_dir=output_dir,
            tts_backend=tts_backend,
        )
        state.local_regen_worker = worker
    return worker


def _build_local_regen_detail(state: _PipelineState, ui_lang: str = "en") -> str | None:
    worker = state.local_regen_worker
    if worker is None or not worker.is_active():
        return None
    current_episode_id, pending_episode_ids, tts_backend = worker.snapshot()
    detail_parts = [_ui_text(ui_lang, "local_regen_detail", tts_backend=tts_backend)]
    if current_episode_id:
        detail_parts.append(_ui_text(ui_lang, "local_regen_current", episode_id=current_episode_id))
    if pending_episode_ids:
        detail_parts.append(_ui_text(ui_lang, "local_regen_pending", episode_ids=", ".join(pending_episode_ids)))
    return " ".join(detail_parts)


def _save_script_text(
    script_path_value: str | None,
    rows_value: object,
    ui_lang: str = "en",
) -> tuple[dict, dict]:
    if not script_path_value:
        return (
            gr.update(value=None),
            gr.update(visible=True, value=_ui_text(ui_lang, "save_requires_selection")),
        )

    script_path = Path(script_path_value)
    if not script_path.is_file():
        return (
            gr.update(value=None),
            gr.update(visible=True, value=_ui_text(ui_lang, "save_missing_path", path=script_path)),
        )

    rows = _normalize_editor_rows(rows_value)
    serialized_text = _serialize_script_editor_rows(rows, script_path_value, ui_lang)
    try:
        _normalize_and_save_script_text(script_path, serialized_text)
    except ValueError:
        return (
            gr.update(value=rows),
            gr.update(visible=True, value=_ui_text(ui_lang, "save_empty_rows")),
        )
    updated_rows, _, _ = _build_script_editor_rows(str(script_path.resolve()), ui_lang)
    return (
        gr.update(value=updated_rows),
        gr.update(visible=True, value=_ui_text(ui_lang, "save_success", path=script_path)),
    )


def _normalize_and_save_script_text(script_path: Path, raw_text: str) -> str:
    normalized_script_text = normalize_text(raw_text)
    if not normalized_script_text.strip():
        raise ValueError("normalized script text is empty")
    save_text(script_path, normalized_script_text)
    return normalized_script_text


def _build_preview_response_with_status(
    response: tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None],
    message: str,
) -> tuple[dict, str, dict, dict, dict, list[list[str]] | None, list[str] | None, dict, str | None]:
    return (
        response[0],
        response[1],
        response[2],
        response[3],
        response[4],
        response[5],
        response[6],
        gr.update(visible=True, value=message),
        response[8],
    )


def overwrite_script_from_uploaded_txt(
    uploaded_script_value: object,
    target_script_path_value: str | None,
    ui_lang: str = "en",
):
    target_script_path = _extract_existing_path(target_script_path_value)
    if not target_script_path:
        return _build_preview_response_with_status(
            _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), None, ui_lang, _ui_text(ui_lang, "preview_script_empty")),
            _ui_text(ui_lang, "target_script_missing"),
        )

    uploaded_script_path = _extract_existing_path(uploaded_script_value)
    if not uploaded_script_path:
        return _build_preview_response_with_status(
            _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), target_script_path, ui_lang, _ui_text(ui_lang, "preview_script_empty")),
            _ui_text(ui_lang, "upload_missing_file"),
        )

    target_path = Path(target_script_path)
    uploaded_path = Path(uploaded_script_path)
    if uploaded_path.suffix.lower() != ".txt":
        return _build_preview_response_with_status(
            _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), str(target_path), ui_lang, _ui_text(ui_lang, "preview_script_empty")),
            _ui_text(ui_lang, "upload_only_txt"),
        )

    try:
        uploaded_text = uploaded_path.read_text(encoding="utf-8", errors="replace")
        _normalize_and_save_script_text(target_path, uploaded_text)
    except ValueError:
        return _build_preview_response_with_status(
            _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), str(target_path), ui_lang, _ui_text(ui_lang, "preview_script_empty")),
            _ui_text(ui_lang, "upload_normalized_empty"),
        )
    except OSError as exc:
        return _build_preview_response_with_status(
            _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), str(target_path), ui_lang, _ui_text(ui_lang, "preview_script_empty")),
            _ui_text(ui_lang, "upload_failed", error=exc),
        )

    return _build_preview_response_with_status(
        _build_text_preview_response(_ui_text(ui_lang, "script_editor_title"), str(target_path), ui_lang, _ui_text(ui_lang, "preview_script_empty")),
        _ui_text(ui_lang, "upload_success", name=target_path.name),
    )


def _terminate_job_process(state: _PipelineState) -> None:
    process = state.process
    if process is None:
        return

    _refresh_process_state(state)
    if state.done:
        return

    state.termination_requested = True
    _emit_state_text(state, "\n[AnyPod UI] Stopping the current run...\n")
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
            _emit_state_text(state, "[AnyPod UI] SIGTERM sent. The job process has exited.\n")
        except subprocess.TimeoutExpired:
            _emit_state_text(state, "[AnyPod UI] The process is still alive after SIGTERM. Sending SIGKILL.\n")
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                _emit_state_text(state, "[AnyPod UI] The process still appears alive after SIGKILL. Check the system process table.\n")
    except ProcessLookupError:
        _emit_state_text(state, "[AnyPod UI] The job process no longer exists.\n")
    finally:
        _refresh_process_state(state)


def start_pipeline_job(
    uploaded_file: str | None,
    input_path_text: str,
    output_dir_text: str,
    dialogue_mode: str,
    primary_language: str,
    mode: str,
    target_episode_level: str,
    tts_backend: str,
    allow_external_knowledge: bool,
    show_title: str,
    positioning: str,
    target_audience: str,
    fixed_opening: str,
    speaker_1_name: str,
    speaker_1_style: str,
    speaker_1_voice_id: str,
    speaker_1_language: str,
    speaker_2_name: str,
    speaker_2_style: str,
    speaker_2_voice_id: str,
    speaker_2_language: str,
    chunk_max_words: float | int | None,
    step2_num_workers: float | int | None,
    max_retries: float | int | None,
    current_job_id: str | None,
    ui_lang: str,
):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    existing_state = _get_job(current_job_id)
    if existing_state is not None and not existing_state.done:
        logs, _ = existing_state.logs.snapshot_and_version()
        output_dir = existing_state.output_dir
        _, script_files, audio_files = _collect_live_outputs(output_dir)
        script_files_update = _maybe_skip_list_update(existing_state.script_files, script_files)
        audio_files_update = _maybe_skip_list_update(existing_state.audio_files, audio_files)
        existing_state.script_files = script_files
        existing_state.audio_files = audio_files
        return _build_running_response(
            output_dir,
            logs,
            script_files_update,
            audio_files_update,
            current_job_id,
            ui_lang=resolved_ui_lang,
        )

    try:
        args = build_runtime_args(
            uploaded_file,
            input_path_text,
            output_dir_text,
            dialogue_mode,
            primary_language,
            mode,
            target_episode_level,
            tts_backend,
            allow_external_knowledge,
            show_title,
            positioning,
            target_audience,
            fixed_opening,
            speaker_1_name,
            speaker_1_style,
            speaker_1_voice_id,
            speaker_1_language,
            speaker_2_name,
            speaker_2_style,
            speaker_2_voice_id,
            speaker_2_language,
            chunk_max_words,
            step2_num_workers,
            max_retries,
        )
    except Exception as exc:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "run_validation_error_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "run_validation_error_detail", error=exc),
            None,
            str(exc),
            None,
            None,
            None,
            ui_lang=resolved_ui_lang,
        )

    state = _PipelineState(output_dir=args.output_dir)
    _emit_state_text(state, f"[AnyPod UI] Job started\nInput: {args.input_path}\nOutput: {args.output_dir}\n\n")
    job_id = uuid.uuid4().hex
    _register_job(job_id, state)
    try:
        _launch_pipeline_process(state, args)
    except Exception as exc:
        state.error_message = str(exc)
        state.done = True
        _cleanup_job_files(state)
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "run_launch_error_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "run_launch_error_detail", error=exc),
            args.output_dir,
            state.logs.snapshot(),
            None,
            None,
            job_id,
            ui_lang=resolved_ui_lang,
        )

    logs, _ = state.logs.snapshot_and_version()
    state.script_files = []
    state.audio_files = []
    return _build_running_response(
        args.output_dir,
        logs,
        [],
        [],
        job_id,
        ui_lang=resolved_ui_lang,
    )


def poll_pipeline_job(
    job_id: str | None,
    ui_lang: str,
):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    state = _get_job(job_id)
    if state is None:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "status_wait_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "status_wait_detail"),
            None,
            "",
            None,
            None,
            None,
            ui_lang=resolved_ui_lang,
        )

    _refresh_process_state(state)
    logs, _ = state.logs.snapshot_and_version()
    output_dir = state.output_dir
    if not state.done:
        _, script_files, audio_files = _collect_live_outputs(output_dir)
        script_files_update = _maybe_skip_list_update(state.script_files, script_files)
        audio_files_update = _maybe_skip_list_update(state.audio_files, audio_files)
        state.script_files = script_files
        state.audio_files = audio_files
        detail = _ui_text(resolved_ui_lang, "run_terminating_detail") if state.termination_requested else _ui_text(
            resolved_ui_lang,
            "run_running_detail",
        )
        return _build_running_response(
            output_dir,
            logs,
            script_files_update,
            audio_files_update,
            job_id,
            ui_lang=resolved_ui_lang,
            detail=detail,
            stop_enabled=not state.termination_requested,
        )

    if output_dir is not None:
        _finalize_job_outputs(state)

    local_regen_detail = _build_local_regen_detail(state, resolved_ui_lang)
    if local_regen_detail is not None:
        _, script_files, audio_files = _collect_live_outputs(output_dir)
        script_files_update = _maybe_skip_list_update(state.script_files, script_files)
        audio_files_update = _maybe_skip_list_update(state.audio_files, audio_files)
        state.script_files = script_files
        state.audio_files = audio_files
        return _build_running_response(
            output_dir,
            logs,
            script_files_update,
            audio_files_update,
            job_id,
            ui_lang=resolved_ui_lang,
            detail=local_regen_detail,
            stop_enabled=False,
        )

    if state.terminated:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "status_terminated_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "run_stopped_detail"),
            output_dir,
            logs,
            state.script_files,
            state.audio_files,
            job_id,
            ui_lang=resolved_ui_lang,
        )

    if state.error_message:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "status_failed_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "run_failed_detail", error=state.error_message),
            output_dir,
            logs,
            state.script_files,
            state.audio_files,
            job_id,
            ui_lang=resolved_ui_lang,
        )

    return _build_finished_response(
        _ui_text(resolved_ui_lang, "status_completed_title").removeprefix("### ").strip(),
        _ui_text(resolved_ui_lang, "run_completed_detail"),
        output_dir,
        logs,
        state.script_files,
        state.audio_files,
        job_id,
        ui_lang=resolved_ui_lang,
    )


def stop_pipeline_job(
    job_id: str | None,
    ui_lang: str,
):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    state = _get_job(job_id)
    if state is None:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "stop_no_job_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "stop_no_job_detail"),
            None,
            "",
            None,
            None,
            None,
            ui_lang=resolved_ui_lang,
        )

    _terminate_job_process(state)
    return poll_pipeline_job(job_id, resolved_ui_lang)


def reset_preview_panel(ui_lang: str = "en"):
    return _empty_preview_response(_normalize_ui_language(ui_lang))


def start_pipeline_job_and_reset_preview(*args):
    ui_lang = args[-1] if args else "en"
    return start_pipeline_job(*args) + reset_preview_panel(ui_lang)


def preview_script_path(
    path_value: str | None,
    ui_lang: str,
):
    return _build_text_preview_response(
        _ui_text(ui_lang, "script_editor_title"),
        path_value,
        ui_lang,
        _ui_text(ui_lang, "preview_script_empty"),
    )


def preview_audio_path(
    path_value: str | None,
    ui_lang: str,
):
    return _build_audio_preview_response(path_value, ui_lang)


def request_audio_regeneration(
    job_id: str | None,
    audio_path_value: str | None,
    ui_lang: str,
):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    state = _get_job(job_id)
    if state is None:
        return _build_finished_response(
            _ui_text(resolved_ui_lang, "regen_no_job_title").removeprefix("### ").strip(),
            _ui_text(resolved_ui_lang, "regen_no_job_detail"),
            None,
            "",
            None,
            None,
            None,
            ui_lang=resolved_ui_lang,
        )

    audio_path = _read_audio_preview(audio_path_value)
    if audio_path is None:
        _emit_state_text(state, "[AnyPod UI] Regeneration failed: audio file does not exist.\n")
        return poll_pipeline_job(job_id, resolved_ui_lang)

    output_dir = _resolve_output_dir_for_preview(audio_path)
    if output_dir is None:
        _emit_state_text(state, f"[AnyPod UI] Regeneration failed: could not resolve the output directory for file={audio_path}\n")
        return poll_pipeline_job(job_id, resolved_ui_lang)

    episode_id = Path(audio_path).stem
    script_path = output_dir / "scripts" / f"{episode_id}.txt"
    if not script_path.is_file():
        _emit_state_text(state, f"[AnyPod UI] Regeneration failed: missing matching script {script_path}\n")
        return poll_pipeline_job(job_id, resolved_ui_lang)

    if (
        not state.done
        and not state.termination_requested
        and state.output_dir is not None
        and state.output_dir.resolve() == output_dir.resolve()
    ):
        queued = enqueue_tts_priority_episode(output_dir, episode_id)
        if queued:
            _emit_state_text(state, f"[AnyPod UI] Inserted {episode_id} into the next priority slot of the current TTS queue.\n")
        else:
            _emit_state_text(state, f"[AnyPod UI] {episode_id} is already present in the current TTS priority queue.\n")
        return poll_pipeline_job(job_id, resolved_ui_lang)

    tts_backend = _resolve_tts_backend_for_episode(output_dir, episode_id, DEFAULT_ARGS.tts_backend)
    worker = _ensure_local_regen_worker(
        state=state,
        output_dir=output_dir,
        tts_backend=tts_backend,
    )
    queue_message = worker.enqueue(episode_id)
    _emit_state_text(state, f"[AnyPod UI] {queue_message}\n")
    return poll_pipeline_job(job_id, resolved_ui_lang)


def _toggle_speaker_2(dialogue_mode: str) -> dict:
    return gr.update(visible=_normalize_dialogue_mode_value(dialogue_mode, DEFAULT_ARGS.dialogue_mode) == "dual")


def load_voice_library_entry(selected_voice_id: str | None, ui_lang: str):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    entry = find_voice_entry(selected_voice_id)
    if entry is None:
        return (
            "",
            "",
            "",
            "",
            gr.update(value=None),
            _ui_text(resolved_ui_lang, "voice_library_status_idle"),
        )

    return (
        entry.get("voice_id", ""),
        entry.get("speaker_name", ""),
        entry.get("prompt_audio", ""),
        entry.get("prompt_text", ""),
        gr.update(value=None),
        _ui_text(resolved_ui_lang, "voice_library_loaded", voice_id=entry.get("voice_id", "")),
    )


def reset_voice_library_editor(ui_lang: str):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    return (
        gr.update(value=None, choices=_voice_choice_pairs()),
        "",
        "",
        "",
        "",
        gr.update(value=None),
        _ui_text(resolved_ui_lang, "voice_library_status_idle"),
    )


def save_voice_library_entry(
    selected_voice_id: str | None,
    voice_id: str,
    speaker_name: str,
    prompt_audio_path: str,
    uploaded_prompt_audio: str | None,
    prompt_text: str,
    current_speaker_1_voice_id: str | None,
    current_speaker_2_voice_id: str | None,
    ui_lang: str,
):
    resolved_ui_lang = _normalize_ui_language(ui_lang)
    previous_voice_id = _optional_text(selected_voice_id)
    resolved_prompt_audio_path = "" if _clean_text(uploaded_prompt_audio) else prompt_audio_path
    try:
        _, saved_entry = upsert_voice_entry(
            voice_id=voice_id,
            speaker_name=speaker_name,
            prompt_text=prompt_text,
            prompt_audio_path=resolved_prompt_audio_path,
            uploaded_prompt_audio_path=uploaded_prompt_audio,
            previous_voice_id=previous_voice_id,
        )
    except Exception as exc:
        return (
            _ui_text(resolved_ui_lang, "voice_library_save_failed", error=exc),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.update(value=None),
            gr.skip(),
            gr.skip(),
        )

    next_speaker_1_voice_id = _clean_text(current_speaker_1_voice_id)
    if previous_voice_id and next_speaker_1_voice_id == previous_voice_id:
        next_speaker_1_voice_id = saved_entry["voice_id"]
    next_speaker_2_voice_id = _clean_text(current_speaker_2_voice_id)
    if previous_voice_id and next_speaker_2_voice_id == previous_voice_id:
        next_speaker_2_voice_id = saved_entry["voice_id"]

    return (
        _ui_text(resolved_ui_lang, "voice_library_saved", voice_id=saved_entry["voice_id"]),
        _build_voice_dropdown_update(
            label=_ui_text(resolved_ui_lang, "voice_library_selector_label"),
            value=saved_entry["voice_id"],
        ),
        saved_entry["voice_id"],
        saved_entry["speaker_name"],
        saved_entry["prompt_audio"],
        saved_entry["prompt_text"],
        gr.update(value=None),
        _build_voice_dropdown_update(
            label=_ui_text(resolved_ui_lang, "voice_id_label"),
            value=next_speaker_1_voice_id or None,
        ),
        _build_voice_dropdown_update(
            label=_ui_text(resolved_ui_lang, "voice_id_label"),
            value=next_speaker_2_voice_id or None,
        ),
    )


def _resolve_vscode_root_path(vscode_proxy_uri: str | None, server_port: int) -> str | None:
    if not vscode_proxy_uri:
        return None

    raw = vscode_proxy_uri.strip()
    if not raw or raw == "/":
        return None

    port_str = str(server_port)
    replacements = (
        "{{port}}",
        "{port}",
        "%7B%7Bport%7D%7D",
        "%7b%7bport%7d%7d",
        "%7Bport%7D",
        "%7bport%7d",
    )

    resolved = raw
    for token in replacements:
        resolved = resolved.replace(token, port_str)

    parsed = urllib.parse.urlsplit(resolved)
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunsplit(parsed)

    if not resolved.startswith("/"):
        resolved = "/" + resolved
    if resolved == "/":
        return None
    return resolved.rstrip("/")


def _build_static_ui_updates(ui_lang: str) -> tuple[dict, ...]:
    lang = _normalize_ui_language(ui_lang)
    content_default = "en" if lang == "en" else "zh"
    return (
        gr.update(value=_ui_text(lang, "app_intro")),
        gr.update(value=_ui_text(lang, "settings_title")),
        gr.update(value=_ui_text(lang, "run_button")),
        gr.update(value=_ui_text(lang, "stop_button")),
        gr.update(value=_ui_text(lang, "input_section")),
        gr.update(label=_ui_text(lang, "uploaded_file_label")),
        gr.update(label=_ui_text(lang, "input_path_label"), placeholder=_ui_text(lang, "input_path_placeholder")),
        gr.update(label=_ui_text(lang, "output_dir_label"), placeholder=_ui_text(lang, "output_dir_placeholder")),
        gr.update(label=_ui_text(lang, "dialogue_mode_label"), choices=_choice_pairs(DIALOGUE_MODE_UI_LABELS, lang)),
        gr.update(
            label=_ui_text(lang, "primary_language_label"),
            choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, lang),
            value=content_default,
        ),
        gr.update(label=_ui_text(lang, "program_mode_label"), choices=_choice_pairs(PROGRAM_MODE_UI_LABELS, lang)),
        gr.update(label=_ui_text(lang, "target_episode_level_label")),
        gr.update(label=_ui_text(lang, "tts_backend_label")),
        gr.update(label=_ui_text(lang, "allow_external_knowledge_label")),
        gr.update(label=_ui_text(lang, "show_title_label"), placeholder=_ui_text(lang, "show_title_placeholder")),
        gr.update(label=_ui_text(lang, "positioning_label"), placeholder=_ui_text(lang, "positioning_placeholder")),
        gr.update(label=_ui_text(lang, "target_audience_label"), placeholder=_ui_text(lang, "target_audience_placeholder")),
        gr.update(label=_ui_text(lang, "fixed_opening_label"), placeholder=_ui_text(lang, "fixed_opening_placeholder")),
        gr.update(value=_ui_text(lang, "speaker_1_title")),
        gr.update(label=_ui_text(lang, "name_label"), placeholder=_ui_text(lang, "speaker_1_name_placeholder")),
        gr.update(label=_ui_text(lang, "style_label"), placeholder=_ui_text(lang, "speaker_1_style_placeholder")),
        _build_voice_dropdown_update(label=_ui_text(lang, "voice_id_label")),
        gr.update(
            label=_ui_text(lang, "language_label"),
            choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, lang),
            value=content_default,
        ),
        gr.update(value=_ui_text(lang, "speaker_2_title")),
        gr.update(label=_ui_text(lang, "name_label"), placeholder=_ui_text(lang, "speaker_2_name_placeholder")),
        gr.update(label=_ui_text(lang, "style_label"), placeholder=_ui_text(lang, "speaker_2_style_placeholder")),
        _build_voice_dropdown_update(label=_ui_text(lang, "voice_id_label")),
        gr.update(
            label=_ui_text(lang, "language_label"),
            choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, lang),
            value=content_default,
        ),
        gr.update(label=_ui_text(lang, "voice_library_title").removeprefix("### ").strip()),
        _build_voice_dropdown_update(label=_ui_text(lang, "voice_library_selector_label")),
        gr.update(label=_ui_text(lang, "voice_id_label")),
        gr.update(label=_ui_text(lang, "voice_library_speaker_name_label")),
        gr.update(label=_ui_text(lang, "voice_library_prompt_audio_path_label")),
        gr.update(value=_ui_text(lang, "or_separator")),
        gr.update(label=_ui_text(lang, "voice_library_prompt_audio_upload_label")),
        gr.update(label=_ui_text(lang, "voice_library_prompt_text_label")),
        gr.update(value=_ui_text(lang, "voice_library_new_button")),
        gr.update(value=_ui_text(lang, "voice_library_save_button")),
        gr.update(value=_ui_text(lang, "status_panel_title")),
        gr.update(label=_ui_text(lang, "logs_label")),
        gr.update(value=_ui_text(lang, "artifact_panel_title")),
        gr.update(value=_ui_text(lang, "artifact_hint")),
        gr.update(value=_ui_text(lang, "scripts_block_title")),
        gr.update(value=_ui_text(lang, "audios_block_title")),
        gr.update(value=_ui_text(lang, "preview_panel_title")),
        gr.update(value=_ui_text(lang, "preview_default_title")),
        gr.update(value=_ui_text(lang, "preview_close_button")),
    )


def _build_ui_language_bridge(ui_lang: str) -> str:
    lang = _normalize_ui_language(ui_lang)
    return (
        "<div aria-hidden=\"true\" style=\"display:none\"></div>"
        "<script>"
        f"window.__anypodUiLang = {json.dumps(lang)};"
        f"document.documentElement.dataset.anypodUiLang = {json.dumps(lang)};"
        "</script>"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AnyPod Gradio") as demo:
        with gr.Row(elem_classes="app-toolbar"):
            with gr.Column(scale=12):
                intro_markdown = gr.Markdown(_ui_text("en", "app_intro"))
            with gr.Column(scale=1, min_width=88, elem_classes="language-switch-wrap"):
                ui_language = gr.Radio(
                    choices=[("EN", "en"), ("ZH", "zh")],
                    value="en",
                    show_label=False,
                    container=False,
                    elem_id="ui-language-switch",
                )
        ui_language_bridge = gr.HTML(_build_ui_language_bridge("en"), visible=False)
        job_state = gr.State(None)
        script_files_state = gr.JSON(value=None, visible=False)
        audio_files_state = gr.JSON(value=None, visible=False)
        script_editor_rows_state = gr.JSON(value=None, visible=False)
        script_speaker_choices_state = gr.JSON(value=None, visible=False)
        current_script_path_state = gr.State(None)
        poll_timer = gr.Timer(POLL_INTERVAL_SEC, active=False)

        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel"):
                settings_title_markdown = gr.Markdown(_ui_text("en", "settings_title"), elem_classes="block-title")
                with gr.Row():
                    run_button = gr.Button(_ui_text("en", "run_button"), variant="primary", elem_classes="run-button")
                    stop_button = gr.Button(_ui_text("en", "stop_button"), variant="stop", interactive=False)

                with gr.Group():
                    input_section_markdown = gr.Markdown(_ui_text("en", "input_section"))
                    uploaded_file = gr.File(
                        label=_ui_text("en", "uploaded_file_label"),
                        file_types=[".pdf", ".txt"],
                        type="filepath",
                    )
                    input_path_text = gr.Textbox(
                        label=_ui_text("en", "input_path_label"),
                        placeholder=_ui_text("en", "input_path_placeholder"),
                    )
                output_dir_text = gr.Textbox(
                    label=_ui_text("en", "output_dir_label"),
                    placeholder=_ui_text("en", "output_dir_placeholder"),
                )

                with gr.Row():
                    dialogue_mode = gr.Dropdown(
                        label=_ui_text("en", "dialogue_mode_label"),
                        choices=_choice_pairs(DIALOGUE_MODE_UI_LABELS, "en"),
                        value=_dialogue_mode_dropdown_value(DEFAULT_ARGS.dialogue_mode),
                    )
                    primary_language = gr.Dropdown(
                        label=_ui_text("en", "primary_language_label"),
                        choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, "en"),
                        value="en",
                    )
                    mode = gr.Dropdown(
                        label=_ui_text("en", "program_mode_label"),
                        choices=_choice_pairs(PROGRAM_MODE_UI_LABELS, "en"),
                        value=_program_mode_dropdown_value(DEFAULT_ARGS.mode),
                    )

                with gr.Row():
                    target_episode_level = gr.Dropdown(
                        label=_ui_text("en", "target_episode_level_label"),
                        choices=["low", "mid", "high"],
                        value=DEFAULT_ARGS.target_episode_level,
                    )
                    tts_backend = gr.Dropdown(
                        label=_ui_text("en", "tts_backend_label"),
                        choices=["moss-ttsd", "vibevoice", "moss-tts", "moss-tts(api)"],
                        value=DEFAULT_ARGS.tts_backend,
                    )
                allow_external_knowledge = gr.Checkbox(
                    label=_ui_text("en", "allow_external_knowledge_label"),
                    value=DEFAULT_ARGS.allow_external_knowledge,
                )
                show_title = gr.Textbox(label=_ui_text("en", "show_title_label"), placeholder=_ui_text("en", "show_title_placeholder"))
                positioning = gr.Textbox(
                    label=_ui_text("en", "positioning_label"),
                    placeholder=_ui_text("en", "positioning_placeholder"),
                )
                target_audience = gr.Textbox(
                    label=_ui_text("en", "target_audience_label"),
                    placeholder=_ui_text("en", "target_audience_placeholder"),
                )
                fixed_opening = gr.Textbox(
                    label=_ui_text("en", "fixed_opening_label"),
                    placeholder=_ui_text("en", "fixed_opening_placeholder"),
                )

                with gr.Group():
                    speaker_1_title_markdown = gr.Markdown(_ui_text("en", "speaker_1_title"))
                    with gr.Row():
                        speaker_1_name = gr.Textbox(label=_ui_text("en", "name_label"), placeholder=_ui_text("en", "speaker_1_name_placeholder"))
                        speaker_1_style = gr.Textbox(
                            label=_ui_text("en", "style_label"),
                            placeholder=_ui_text("en", "speaker_1_style_placeholder"),
                        )
                    with gr.Row():
                        speaker_1_voice_id = gr.Dropdown(
                            label=_ui_text("en", "voice_id_label"),
                            choices=_voice_choice_pairs(),
                            value=_resolve_voice_dropdown_value(DEFAULT_ARGS.speaker_1_voice_id),
                            interactive=True,
                        )
                        speaker_1_language = gr.Dropdown(
                            label=_ui_text("en", "language_label"),
                            choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, "en"),
                            value="en",
                        )

                with gr.Column(visible=DEFAULT_ARGS.dialogue_mode == "dual") as speaker_2_group:
                    speaker_2_title_markdown = gr.Markdown(_ui_text("en", "speaker_2_title"))
                    with gr.Row():
                        speaker_2_name = gr.Textbox(label=_ui_text("en", "name_label"), placeholder=_ui_text("en", "speaker_2_name_placeholder"))
                        speaker_2_style = gr.Textbox(
                            label=_ui_text("en", "style_label"),
                            placeholder=_ui_text("en", "speaker_2_style_placeholder"),
                        )
                    with gr.Row():
                        speaker_2_voice_id = gr.Dropdown(
                            label=_ui_text("en", "voice_id_label"),
                            choices=_voice_choice_pairs(),
                            value=_resolve_voice_dropdown_value(DEFAULT_ARGS.speaker_2_voice_id),
                            interactive=True,
                        )
                        speaker_2_language = gr.Dropdown(
                            label=_ui_text("en", "language_label"),
                            choices=_choice_pairs(CONTENT_LANGUAGE_UI_LABELS, "en"),
                            value="en",
                        )

                with gr.Accordion(_ui_text("en", "voice_library_title").removeprefix("### ").strip(), open=False) as voice_library_title_markdown:
                    voice_library_selector = gr.Dropdown(
                        label=_ui_text("en", "voice_library_selector_label"),
                        choices=_voice_choice_pairs(),
                        value=None,
                        interactive=True,
                    )
                    voice_library_voice_id = gr.Textbox(label=_ui_text("en", "voice_id_label"))
                    voice_library_speaker_name = gr.Textbox(label=_ui_text("en", "voice_library_speaker_name_label"))
                    voice_library_prompt_audio_path = gr.Textbox(label=_ui_text("en", "voice_library_prompt_audio_path_label"))
                    voice_library_audio_or_markdown = gr.Markdown(_ui_text("en", "or_separator"))
                    voice_library_prompt_audio_upload = gr.File(
                        label=_ui_text("en", "voice_library_prompt_audio_upload_label"),
                        file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
                        type="filepath",
                    )
                    voice_library_prompt_text = gr.Textbox(
                        label=_ui_text("en", "voice_library_prompt_text_label"),
                        lines=4,
                    )
                    with gr.Row():
                        voice_library_new_button = gr.Button(_ui_text("en", "voice_library_new_button"))
                        voice_library_save_button = gr.Button(_ui_text("en", "voice_library_save_button"), variant="secondary")
                    voice_library_status = gr.Markdown(_ui_text("en", "voice_library_status_idle"))

                with gr.Accordion(_ui_text("en", "advanced_options_label"), open=False):
                    with gr.Row():
                        chunk_max_words = gr.Number(
                            label="chunk_max_words",
                            value=DEFAULT_ARGS.chunk_max_words,
                            precision=0,
                        )
                        step2_num_workers = gr.Number(
                            label="step2_num_workers",
                            value=GRADIO_DEFAULT_STEP2_NUM_WORKERS,
                            precision=0,
                        )
                        max_retries = gr.Number(
                            label="max_retries",
                            value=GRADIO_DEFAULT_MAX_RETRIES,
                            precision=0,
                        )

            with gr.Column(scale=3, elem_classes="panel"):
                status_panel_title_markdown = gr.Markdown(_ui_text("en", "status_panel_title"), elem_classes="block-title")
                status_markdown = gr.Markdown(f"{_ui_text('en', 'status_wait_title')}\n\n{_ui_text('en', 'status_wait_detail')}")
                logs = gr.Textbox(label=_ui_text("en", "logs_label"), lines=24, max_lines=32, interactive=False)

            with gr.Column(scale=3, elem_classes=["panel", "artifact-panel"]):
                artifact_panel_title_markdown = gr.Markdown(_ui_text("en", "artifact_panel_title"), elem_classes="block-title")
                artifact_hint_markdown = gr.Markdown(_ui_text("en", "artifact_hint"))
                with gr.Column(elem_classes="artifact-block"):
                    script_block_title_markdown = gr.Markdown(_ui_text("en", "scripts_block_title"), elem_classes="artifact-block-title")
                    with gr.Column(elem_classes="artifact-list-scroll") as script_entries_container:
                        pass

                with gr.Column(elem_classes="artifact-block"):
                    audio_block_title_markdown = gr.Markdown(_ui_text("en", "audios_block_title"), elem_classes="artifact-block-title")
                    with gr.Column(elem_classes="artifact-list-scroll") as audio_entries_container:
                        pass

            with gr.Column(scale=4, elem_classes=["panel", "preview-panel"], visible=False) as preview_panel:
                preview_panel_title_markdown = gr.Markdown(_ui_text("en", "preview_panel_title"), elem_classes="block-title")
                with gr.Row():
                    preview_title = gr.Markdown(_ui_text("en", "preview_default_title"))
                    preview_download = gr.DownloadButton(
                        _ui_text("en", "preview_download_button"),
                        visible=False,
                        size="sm",
                    )
                    close_preview_button = gr.Button(_ui_text("en", "preview_close_button"), size="sm")
                audio_preview = gr.HTML(
                    value="",
                    label=_ui_text("en", "audio_preview_title"),
                    html_template=AUDIO_PREVIEW_HTML_TEMPLATE,
                    css_template=AUDIO_PREVIEW_CSS_TEMPLATE,
                    js_on_load=AUDIO_PREVIEW_JS_ON_LOAD,
                    apply_default_css=False,
                    visible=False,
                    elem_classes=["anypod-audio-preview-root", "preview-audio-fixed"],
                    container=False,
                    padding=False,
                )
                with gr.Column(elem_classes="preview-scroll"):
                    with gr.Column(visible=False, elem_classes="script-editor-panel") as script_editor_panel:
                        with gr.Column() as script_turns_container:
                            pass
                        script_save_status = gr.Markdown(visible=False)

        with script_entries_container:
            @gr.render(inputs=[script_files_state, ui_language])
            def render_script_entries(script_paths: list[str] | None, current_ui_lang: str):
                lang = _normalize_ui_language(current_ui_lang)
                if not script_paths:
                    gr.Markdown(_ui_text(lang, "no_generated_scripts"))
                    return

                for raw_path in script_paths:
                    path_value = _extract_existing_path(raw_path) or _extract_path_value(raw_path)
                    if not path_value:
                        continue
                    file_name = Path(path_value).name
                    row_key = ("script-entry", path_value)
                    with gr.Row(elem_classes="artifact-entry", key=row_key):
                        preview_button = gr.Button(
                            f"[TXT] {file_name}",
                            variant="secondary",
                            scale=6,
                            elem_classes="artifact-file-button",
                            key=("script-entry-preview-button", path_value),
                        )
                        gr.DownloadButton(
                            _ui_text(lang, "download_button"),
                            value=path_value,
                            size="sm",
                            scale=2,
                            elem_classes="artifact-action-button",
                            key=("script-entry-download-button", path_value),
                        )
                        upload_button = gr.UploadButton(
                            _ui_text(lang, "upload_override_button"),
                            size="sm",
                            scale=3,
                            type="filepath",
                            file_count="single",
                            file_types=[".txt"],
                            elem_classes="artifact-action-button",
                            key=("script-entry-upload-button", path_value),
                        )
                    preview_button.click(
                        fn=lambda lang_value, path=path_value: preview_script_path(path, lang_value),
                        inputs=[ui_language],
                        outputs=[
                            preview_panel,
                            preview_title,
                            preview_download,
                            audio_preview,
                            script_editor_panel,
                            script_editor_rows_state,
                            script_speaker_choices_state,
                            script_save_status,
                            current_script_path_state,
                        ],
                        queue=False,
                        key=("script-entry-preview-event", path_value),
                    )
                    upload_button.upload(
                        fn=lambda uploaded_path, lang_value, target_path=path_value: overwrite_script_from_uploaded_txt(
                            uploaded_path,
                            target_path,
                            lang_value,
                        ),
                        inputs=[upload_button, ui_language],
                        outputs=[
                            preview_panel,
                            preview_title,
                            preview_download,
                            audio_preview,
                            script_editor_panel,
                            script_editor_rows_state,
                            script_speaker_choices_state,
                            script_save_status,
                            current_script_path_state,
                        ],
                        queue=False,
                        key=("script-entry-upload-event", path_value),
                    )

        with script_turns_container:
            @gr.render(inputs=[script_editor_rows_state, script_speaker_choices_state, current_script_path_state, ui_language])
            def render_script_turn_editor(
                rows_value: object,
                speaker_choices_value: object,
                script_path_value: str | None,
                current_ui_lang: str,
            ):
                lang = _normalize_ui_language(current_ui_lang)
                rows, speaker_choices = _normalize_script_editor_state(rows_value, speaker_choices_value)
                row_count = len(rows)
                key_prefix = script_path_value or "no-script-selected"

                if not script_path_value:
                    gr.Markdown(_ui_text(lang, "preview_click_hint"), elem_classes="script-turn-empty")
                    return

                if not rows:
                    gr.Markdown(_ui_text(lang, "preview_no_turns"), elem_classes="script-turn-empty")
                    insert_first_button = gr.Button(
                        _ui_text(lang, "insert_first_turn"),
                        elem_classes="script-insert-button",
                        key=("script-editor-insert-first-button", key_prefix),
                    )
                    insert_first_button.click(
                        fn=lambda speaker_choices: _insert_script_editor_row_from_components(0, speaker_choices, 0),
                        inputs=[script_speaker_choices_state],
                        outputs=[script_editor_rows_state],
                        queue=False,
                        key=("script-editor-insert-first-event", key_prefix),
                    )
                    return

                speaker_components = []
                content_components = []
                insert_before_buttons: list[tuple[gr.Button, int]] = []
                delete_buttons: list[tuple[gr.Button, int]] = []

                for row_index, (speaker_label, content) in enumerate(rows):
                    insert_before_button = gr.Button(
                        _ui_text(lang, "insert_turn"),
                        size="sm",
                        elem_classes="script-insert-button",
                        key=("script-editor-insert-before-button", key_prefix, row_index),
                    )
                    with gr.Column(
                        elem_classes="script-turn-row",
                        key=("script-editor-row", key_prefix, row_index),
                    ):
                        with gr.Row(elem_classes="script-turn-header"):
                            speaker_input = gr.Dropdown(
                                choices=speaker_choices,
                                value=speaker_label,
                                allow_custom_value=False,
                                label=None,
                                show_label=False,
                                scale=3,
                                elem_classes="script-turn-speaker",
                                key=("script-editor-speaker", key_prefix, row_index),
                                interactive=True,
                                preserved_by_key=["value"],
                            )
                            delete_button = gr.Button(
                                _ui_text(lang, "delete_turn"),
                                size="sm",
                                scale=1,
                                variant="stop",
                                elem_classes="script-turn-delete",
                                key=("script-editor-delete-button", key_prefix, row_index),
                            )
                        content_input = gr.Textbox(
                            value=content,
                            label=None,
                            show_label=False,
                            lines=3,
                            elem_classes="script-turn-content",
                            key=("script-editor-content", key_prefix, row_index),
                            interactive=True,
                            preserved_by_key=["value"],
                        )
                    speaker_components.append(speaker_input)
                    content_components.append(content_input)
                    insert_before_buttons.append((insert_before_button, row_index))
                    delete_buttons.append((delete_button, row_index))

                row_component_inputs = [script_speaker_choices_state, *speaker_components, *content_components]

                for button, insert_index in insert_before_buttons:
                    button.click(
                        fn=lambda speaker_choices, *component_values, current_row_count=row_count, target_index=insert_index: _insert_script_editor_row_from_components(
                            current_row_count,
                            speaker_choices,
                            target_index,
                            *component_values,
                        ),
                        inputs=row_component_inputs,
                        outputs=[script_editor_rows_state],
                        queue=False,
                        key=("script-editor-insert-before-event", key_prefix, insert_index),
                    )

                for button, delete_index in delete_buttons:
                    button.click(
                        fn=lambda speaker_choices, *component_values, current_row_count=row_count, target_index=delete_index: _delete_script_editor_row_from_components(
                            current_row_count,
                            speaker_choices,
                            target_index,
                            *component_values,
                        ),
                        inputs=row_component_inputs,
                        outputs=[script_editor_rows_state],
                        queue=False,
                        key=("script-editor-delete-event", key_prefix, delete_index),
                    )

                with gr.Row():
                    add_script_turn_button = gr.Button(
                        _ui_text(lang, "append_turn"),
                        elem_classes="script-insert-button",
                        key=("script-editor-insert-tail-button", key_prefix),
                    )
                    script_save_button = gr.Button(
                        _ui_text(lang, "save_script"),
                        variant="primary",
                        key=("script-editor-save-button", key_prefix),
                    )

                add_script_turn_button.click(
                    fn=lambda speaker_choices, *component_values, current_row_count=row_count: _insert_script_editor_row_from_components(
                        current_row_count,
                        speaker_choices,
                        current_row_count,
                        *component_values,
                    ),
                    inputs=[script_speaker_choices_state, *speaker_components, *content_components],
                    outputs=[script_editor_rows_state],
                    queue=False,
                    key=("script-editor-insert-tail-event", key_prefix),
                )
                script_save_button.click(
                    fn=lambda script_path, speaker_choices, *component_values, current_row_count=row_count: _save_script_text_from_components(
                        script_path,
                        speaker_choices,
                        lang,
                        current_row_count,
                        *component_values,
                    ),
                    inputs=[current_script_path_state, script_speaker_choices_state, *speaker_components, *content_components],
                    outputs=[script_editor_rows_state, script_save_status],
                    queue=False,
                    key=("script-editor-save-event", key_prefix),
                )

        with audio_entries_container:
            @gr.render(inputs=[audio_files_state, ui_language])
            def render_audio_entries(audio_paths: list[str] | None, current_ui_lang: str):
                lang = _normalize_ui_language(current_ui_lang)
                if not audio_paths:
                    gr.Markdown(_ui_text(lang, "no_generated_audio"))
                    return

                for raw_path in audio_paths:
                    path_value = _extract_existing_path(raw_path) or _extract_path_value(raw_path)
                    if not path_value:
                        continue
                    file_name = Path(path_value).name
                    with gr.Row(elem_classes="artifact-entry", key=("audio-entry", path_value)):
                        preview_button = gr.Button(
                            f"[WAV] {file_name}",
                            variant="secondary",
                            scale=7,
                            elem_classes="artifact-file-button",
                            key=("audio-entry-preview-button", path_value),
                        )
                        gr.DownloadButton(
                            _ui_text(lang, "download_button"),
                            value=path_value,
                            size="sm",
                            scale=2,
                            elem_classes="artifact-action-button",
                            key=("audio-entry-download-button", path_value),
                        )
                        regenerate_button = gr.Button(
                            _ui_text(lang, "regenerate_button"),
                            size="sm",
                            scale=2,
                            elem_classes="artifact-action-button",
                            key=("audio-entry-regenerate-button", path_value),
                        )
                    preview_button.click(
                        fn=lambda lang_value, path=path_value: preview_audio_path(path, lang_value),
                        inputs=[ui_language],
                        outputs=[
                            preview_panel,
                            preview_title,
                            preview_download,
                            audio_preview,
                            script_editor_panel,
                            script_editor_rows_state,
                            script_speaker_choices_state,
                            script_save_status,
                            current_script_path_state,
                        ],
                        queue=False,
                        key=("audio-entry-preview-event", path_value),
                    )
                    regenerate_button.click(
                        fn=lambda current_job_id, lang_value, path=path_value: request_audio_regeneration(current_job_id, path, lang_value),
                        inputs=[job_state, ui_language],
                        outputs=[
                            status_markdown,
                            logs,
                            script_files_state,
                            audio_files_state,
                            job_state,
                            poll_timer,
                            run_button,
                            stop_button,
                        ],
                        queue=False,
                        key=("audio-entry-regenerate-event", path_value),
                    )

        dialogue_mode.change(_toggle_speaker_2, inputs=[dialogue_mode], outputs=[speaker_2_group])
        voice_library_selector.change(
            fn=load_voice_library_entry,
            inputs=[voice_library_selector, ui_language],
            outputs=[
                voice_library_voice_id,
                voice_library_speaker_name,
                voice_library_prompt_audio_path,
                voice_library_prompt_text,
                voice_library_prompt_audio_upload,
                voice_library_status,
            ],
            queue=False,
        )
        voice_library_new_button.click(
            fn=reset_voice_library_editor,
            inputs=[ui_language],
            outputs=[
                voice_library_selector,
                voice_library_voice_id,
                voice_library_speaker_name,
                voice_library_prompt_audio_path,
                voice_library_prompt_text,
                voice_library_prompt_audio_upload,
                voice_library_status,
            ],
            queue=False,
        )
        voice_library_save_button.click(
            fn=save_voice_library_entry,
            inputs=[
                voice_library_selector,
                voice_library_voice_id,
                voice_library_speaker_name,
                voice_library_prompt_audio_path,
                voice_library_prompt_audio_upload,
                voice_library_prompt_text,
                speaker_1_voice_id,
                speaker_2_voice_id,
                ui_language,
            ],
            outputs=[
                voice_library_status,
                voice_library_selector,
                voice_library_voice_id,
                voice_library_speaker_name,
                voice_library_prompt_audio_path,
                voice_library_prompt_text,
                voice_library_prompt_audio_upload,
                speaker_1_voice_id,
                speaker_2_voice_id,
            ],
            queue=False,
        )
        close_preview_button.click(
            fn=reset_preview_panel,
            inputs=[ui_language],
            outputs=[
                preview_panel,
                preview_title,
                preview_download,
                audio_preview,
                script_editor_panel,
                script_editor_rows_state,
                script_speaker_choices_state,
                script_save_status,
                current_script_path_state,
            ],
            queue=False,
        )

        run_button.click(
            fn=start_pipeline_job_and_reset_preview,
            inputs=[
                uploaded_file,
                input_path_text,
                output_dir_text,
                dialogue_mode,
                primary_language,
                mode,
                target_episode_level,
                tts_backend,
                allow_external_knowledge,
                show_title,
                positioning,
                target_audience,
                fixed_opening,
                speaker_1_name,
                speaker_1_style,
                speaker_1_voice_id,
                speaker_1_language,
                speaker_2_name,
                speaker_2_style,
                speaker_2_voice_id,
                speaker_2_language,
                chunk_max_words,
                step2_num_workers,
                max_retries,
                job_state,
                ui_language,
            ],
            outputs=[
                status_markdown,
                logs,
                script_files_state,
                audio_files_state,
                job_state,
                poll_timer,
                run_button,
                stop_button,
                preview_panel,
                preview_title,
                preview_download,
                audio_preview,
                script_editor_panel,
                script_editor_rows_state,
                script_speaker_choices_state,
                script_save_status,
                current_script_path_state,
            ],
            queue=False,
        )

        poll_timer.tick(
            fn=poll_pipeline_job,
            inputs=[job_state, ui_language],
            outputs=[
                status_markdown,
                logs,
                script_files_state,
                audio_files_state,
                job_state,
                poll_timer,
                run_button,
                stop_button,
            ],
            queue=False,
        )

        stop_button.click(
            fn=stop_pipeline_job,
            inputs=[job_state, ui_language],
            outputs=[
                status_markdown,
                logs,
                script_files_state,
                audio_files_state,
                job_state,
                poll_timer,
                run_button,
                stop_button,
            ],
            queue=False,
        )

        ui_language.change(
            fn=_build_static_ui_updates,
            inputs=[ui_language],
            outputs=[
                intro_markdown,
                settings_title_markdown,
                run_button,
                stop_button,
                input_section_markdown,
                uploaded_file,
                input_path_text,
                output_dir_text,
                dialogue_mode,
                primary_language,
                mode,
                target_episode_level,
                tts_backend,
                allow_external_knowledge,
                show_title,
                positioning,
                target_audience,
                fixed_opening,
                speaker_1_title_markdown,
                speaker_1_name,
                speaker_1_style,
                speaker_1_voice_id,
                speaker_1_language,
                speaker_2_title_markdown,
                speaker_2_name,
                speaker_2_style,
                speaker_2_voice_id,
                speaker_2_language,
                voice_library_title_markdown,
                voice_library_selector,
                voice_library_voice_id,
                voice_library_speaker_name,
                voice_library_prompt_audio_path,
                voice_library_audio_or_markdown,
                voice_library_prompt_audio_upload,
                voice_library_prompt_text,
                voice_library_new_button,
                voice_library_save_button,
                status_panel_title_markdown,
                logs,
                artifact_panel_title_markdown,
                artifact_hint_markdown,
                script_block_title_markdown,
                audio_block_title_markdown,
                preview_panel_title_markdown,
                preview_title,
                close_preview_button,
            ],
            queue=False,
        )
        ui_language.change(
            fn=_build_ui_language_bridge,
            inputs=[ui_language],
            outputs=[ui_language_bridge],
            queue=False,
        )
        ui_language.change(
            fn=poll_pipeline_job,
            inputs=[job_state, ui_language],
            outputs=[
                status_markdown,
                logs,
                script_files_state,
                audio_files_state,
                job_state,
                poll_timer,
                run_button,
                stop_button,
            ],
            queue=False,
        )

    return demo


def build_launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anypod_gradio",
        description="Launch the AnyPod Gradio interface",
    )
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Gradio bind address")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Enable a Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", help="Open the browser automatically after launch")
    return parser


def main() -> None:
    args = build_launch_parser().parse_args()
    demo = build_demo()
    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = _resolve_vscode_root_path(vscode_proxy_uri, args.server_port)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        root_path=root_path,
        allowed_paths=[str(path) for path in ALLOWED_RETURN_ROOTS],
        show_error=True,
        theme=APP_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
