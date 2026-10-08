"""Separate attachment metadata from visible user text and model instructions."""
import json
import re


ATTACHMENT_NOTICE = "\n\n附件（已保存到工作区，请使用文件工具读取）：\n"


def model_user_text(data):
    text = data.get("text")
    files = data.get("attachments")
    if files:
        return (text or "请查看上传的文件。") + ATTACHMENT_NOTICE + json.dumps(files, ensure_ascii=False)
    return text


def user_message_view(data):
    text = data.get("text") or ""
    files = data.get("attachments") or []
    # Older versions persisted the upload hint inside text. Only recognize the
    # exact generated suffix with valid upload paths; leave other prose intact.
    if "attachments" not in data and ATTACHMENT_NOTICE in text:
        original, _, suffix = text.rpartition(ATTACHMENT_NOTICE)
        try:
            legacy = json.loads(suffix)
            if (isinstance(legacy, list) and legacy and all(
                    isinstance(f, dict) and isinstance(f.get("name"), str)
                    and isinstance(f.get("size"), int) and not isinstance(f["size"], bool)
                    and f["size"] >= 0 and isinstance(f.get("path"), str)
                    and re.search(r"/\.mini-harness/uploads/[0-9a-f]{32}/[^/]+$", f["path"])
                    and f["path"].rsplit("/", 1)[-1] == f["name"] for f in legacy)):
                text, files = original, legacy
        except (ValueError, TypeError):
            pass
    return {"text": text, "attachments": [
        {"name": f["name"], "size": f.get("size", 0)}
        for f in files if isinstance(f, dict) and isinstance(f.get("name"), str)]}


def user_message_title(data):
    view = user_message_view(data)
    text = view["text"] or "、".join(f["name"] for f in view["attachments"])
    return " ".join(text.split())[:80]
