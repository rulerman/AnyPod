from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)

    remove_chars = "【】《》（）『』「」" '"-_“”～~‘’'

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts: list[dict[str, str]] = []

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        match = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = match.groups() if match else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        content = content.replace("——", "，")
        content = content.replace("……", "，")
        content = content.replace("...", "，")
        content = content.replace("⸺", "，")
        content = content.replace("―", "，")
        content = content.replace("—", "，")
        content = content.replace("…", "，")

        internal_punct_map = str.maketrans({"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"})
        content = content.translate(internal_punct_map)
        content = content.strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)

        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("。", "，")
            content = body + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        normalized_text = ""
    else:
        merged_lines: list[str] = []
        current_tag = processed_parts[0]["tag"]
        current_content = [processed_parts[0]["content"]]

        for part in processed_parts[1:]:
            if part["tag"] == current_tag and current_tag:
                current_content.append(part["content"])
            else:
                merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
                current_tag = part["tag"]
                current_content = [part["content"]]

        merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
        normalized_text = "".join(merged_lines).replace("‘", "'").replace("’", "'")

    if not normalized_text.startswith("[S1]"):
        normalized_text = f"[S1]{normalized_text}"
    return normalized_text
