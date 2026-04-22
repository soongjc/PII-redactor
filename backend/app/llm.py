"""Ollama-backed LLM clients for vision OCR (Qwen2.5-VL) and PII tagging (Qwen3)."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ._logging import get_logger
from .config import settings

logger = get_logger("pii.llm")

def _chat_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        settings.ollama_timeout_s, connect=settings.ollama_connect_timeout_s
    )


PII_TYPES = [
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SSN_NRIC",
    "DOB",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "IP",
]


VL_PROMPT_QWEN = """Extract every visible text region from the image.

Return ONLY a JSON object in this exact shape:
{"regions": [{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}, ...]}

Rules:
- bbox_2d uses pixel coordinates of the input image (NOT normalized): x1,y1 is the top-left corner, x2,y2 is the bottom-right corner.
- text_content is the text inside that box, verbatim (preserve casing, punctuation, spacing).
- Group text into the smallest semantically meaningful units (a name, a phone number, an email, a single line of an address, a date). Do not merge unrelated lines.
- SKIP pure monetary amounts and prices (e.g. "233.50", "1,234.56", "RM 1,200", "$100.00", "€50") — they are not PII and are not useful to extract.
- DO extract phone numbers, IDs, account numbers, reference numbers, and dates — even if they contain only digits.
- If the image has no text, return {"regions": []}.
- Do NOT include any prose, markdown, code fences, or explanations. JSON only.
"""


# dots.ocr's native prompt from rednote-hilab/dots.ocr/dots_ocr/utils/prompts.py.
# Output is an array of {bbox: [x1,y1,x2,y2], category, text} where bbox is xyxy.
VL_PROMPT_DOTSOCR = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def _vl_prompt() -> str:
    base = VL_PROMPT_DOTSOCR if settings.vl_prompt_mode == "dotsocr" else VL_PROMPT_QWEN
    # Qwen3-family recognises `/no_think` at the start of a user message and
    # skips reasoning for that turn. Belt-and-braces with the `think: false`
    # option, which some Ollama builds silently ignore. Non-Qwen3 models treat
    # it as an unremarkable prefix.
    if settings.vl_disable_think:
        return "/no_think\n\n" + base
    return base


# Categories from dots.ocr we should never treat as redactable text.
_DOTSOCR_SKIP_CATEGORIES = {"Picture"}


# Match pure money-like strings. Kept narrow so we don't accidentally filter
# phone/account/SSN/date shapes (which don't have decimal points or currency
# prefixes/suffixes). Covers:
#   - Currency-prefixed:  $100, RM 1,234.56, €50.00, USD 10
#   - Currency-suffixed:  1,234.56 MYR, 100 USD
#   - Plain decimal:      233.50, 1,234.56
_MONEY_RX = re.compile(
    r"""^\s*
        (?:
            # Prefix:  $100 / RM 1,234.56 / USD 10
            (?:RM|MYR|USD|SGD|HKD|EUR|GBP|JPY|CNY|\$|€|£|¥|HK\$|S\$|US\$|AU\$|NZ\$)
            \s*[\-+]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?
            |
            # Suffix:  1,234.56 MYR / 100 USD
            [\-+]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?
            \s*(?:RM|MYR|USD|SGD|HKD|EUR|GBP|JPY|CNY|\$|€|£|¥)
            |
            # Bare decimal with 1-2 dp:  233.50 / 1,234.5
            [\-+]?\d{1,3}(?:,\d{3})*\.\d{1,2}
        )
        \s*$
    """,
    re.VERBOSE,
)


def _looks_like_money(text: str) -> bool:
    return bool(_MONEY_RX.match(text))


# /no_think disables Qwen3's internal chain-of-thought which otherwise adds a lot of latency.
PII_PROMPT_TEMPLATE = """/no_think
You are a PII detection engine. Identify Personally Identifiable Information in the input text.

Allowed PII types (use these exact strings):
NAME, EMAIL, PHONE, ADDRESS, SSN_NRIC, DOB, CREDIT_CARD, BANK_ACCOUNT, IP

Return ONLY a JSON object with this exact shape:
{{"entities": [{{"type": "TYPE", "text": "exact substring from input"}}, ...]}}

Rules:
- "text" MUST be an exact substring copied verbatim from the input (same casing/punctuation/whitespace).
- Do NOT infer or invent values that are not present.
- If nothing is found, return {{"entities": []}}.
- No prose, no markdown, no code fences. JSON only.

Input text:
---
{text}
---
"""


def _strip_to_json(raw: str) -> str:
    """Best-effort: strip code fences / surrounding text and return the JSON object substring."""
    s = raw.strip()
    # Drop Qwen3 <think>...</think> blocks if they somehow sneak through.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return s


def _repair_truncated_json(s: str) -> str:
    """Best-effort fix for JSON that ran out of token budget mid-stream.

    Walk the string tracking string/escape state and a brace/bracket stack;
    truncate to the last successful close (the latest `}` or `]` that matched
    its opener), then append closers for whatever stayed unclosed. Drops the
    trailing incomplete element instead of failing the whole page.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    last_complete = -1
    for i, c in enumerate(s):
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c == "}" or c == "]":
            if stack and stack[-1] == c:
                stack.pop()
                last_complete = i
            else:
                # Mismatched close — give up, return original.
                return s
    if last_complete < 0:
        return s
    truncated = s[: last_complete + 1].rstrip().rstrip(",").rstrip()
    # Re-walk the truncated portion to compute the *new* unclosed stack.
    stack = []
    in_str = False
    escape = False
    for c in truncated:
        if escape:
            escape = False; continue
        if c == "\\" and in_str:
            escape = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c == "}" or c == "]":
            if stack and stack[-1] == c:
                stack.pop()
    return truncated + "".join(reversed(stack))


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    stripped = _strip_to_json(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(stripped)
        if repaired != stripped:
            logger.warning("[JSON] repaired truncated output (-> %d chars)", len(repaired))
        return json.loads(repaired)


def _preview(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"... (+{len(s) - n} chars)"


class LLMError(RuntimeError):
    """Raised when the upstream LLM call fails in a way we can display."""


def _classify_error(model: str, body: str) -> str:
    body_l = body.lower()
    # Check crashes FIRST — the body often contains a URL like
    # "llama.cpp/pull/17869" which would otherwise match a naive "pull" hint.
    if any(s in body for s in ("GGML_ASSERT", "SIGABRT", "SIGSEGV", "panic")) or "out of memory" in body_l:
        return (
            "Ollama/llama.cpp runtime crashed (often a Metal bug fixed upstream). "
            "Fixes in order of likelihood: "
            "1) update Ollama (`brew upgrade ollama` or redownload the app), "
            "2) lower VL_NUM_CTX (e.g. 4096), "
            "3) lower VL_MAX_PIXELS (e.g. 602112), "
            "4) try `OLLAMA_FLASH_ATTENTION=0 ollama serve`, "
            f"5) re-pull: `ollama pull {model}`."
        )
    if "does not support images" in body_l or "not a multimodal" in body_l:
        return (
            f"Model '{model}' is not multimodal. Set VL_MODEL to a vision tag "
            f"(e.g. qwen2.5vl:7b, llama3.2-vision:11b, minicpm-v)."
        )
    if "try pulling" in body_l or "model not found" in body_l or "no such model" in body_l:
        return f"Model '{model}' isn't pulled. Run: ollama pull {model}"
    return "See Ollama server logs for details."


def _post_chat_openai(payload: dict[str, Any], *, label: str, base_url: str) -> tuple[dict[str, Any], float]:
    """POST to /v1/chat/completions (llama-server / OpenAI-compatible) with
    streaming SSE. Returns the same dict shape as _post_chat_ollama: {message:{content:str}}."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    model = payload.get("model")
    msg = f"[{label}] -> {url} model={model} (openai)"
    logger.info(msg)
    print(msg, flush=True)

    stream_payload = {**payload, "stream": True}

    t0 = time.time()
    accumulated: list[str] = []
    first_token_at: float | None = None

    try:
        with httpx.Client(timeout=_chat_timeout()) as client:
            with client.stream("POST", url, json=stream_payload) as resp:
                if resp.status_code >= 400:
                    body_bytes = resp.read()
                    body = body_bytes.decode("utf-8", errors="replace")[:500]
                    err_msg = f"[{label}] llama-server returned {resp.status_code}: {body}"
                    logger.warning(err_msg)
                    print(err_msg, flush=True)
                    raise LLMError(
                        f"llama-server {resp.status_code} from '{model}'. Body: {body}"
                    )

                print(f"[{label}] stream> ", end="", flush=True)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # SSE wraps each chunk as "data: {...}" with "[DONE]" to terminate.
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                    else:
                        data_str = line.strip()
                    if data_str == "[DONE]":
                        break
                    if not data_str:
                        continue
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        print(f"\n[{label}] non-json line: {data_str[:200]}", flush=True)
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content") or ""
                    # Some llama-server builds put full message instead of delta:
                    if not chunk and "message" in choices[0]:
                        chunk = choices[0]["message"].get("content", "") or ""
                    if chunk:
                        if first_token_at is None:
                            first_token_at = time.time()
                        print(chunk, end="", flush=True)
                        accumulated.append(chunk)
                print(flush=True)
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot reach llama-server at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise LLMError(f"llama-server request timed out: {e}") from e

    elapsed = time.time() - t0
    content = "".join(accumulated)
    ttft = f", first-token {first_token_at - t0:.2f}s" if first_token_at else ""
    done_msg = f"[{label}] <- {elapsed:.2f}s{ttft} | {len(content)} chars"
    logger.info(done_msg)
    print(done_msg, flush=True)
    return {"message": {"role": "assistant", "content": content}}, elapsed


def _post_chat(payload: dict[str, Any], *, label: str) -> tuple[dict[str, Any], float]:
    """POST to /api/chat with streaming so token output is visible in the terminal
    as the model generates. Accumulates chunks into the same dict shape the caller
    expects from a non-streaming response."""
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    model = payload.get("model")
    msg = f"[{label}] -> {url} model={model}"
    logger.info(msg)
    print(msg, flush=True)

    # Force streaming mode regardless of what the caller put in payload.
    stream_payload = {**payload, "stream": True}

    t0 = time.time()
    accumulated: list[str] = []
    final_obj: dict[str, Any] | None = None
    lines_seen = 0
    first_token_at: float | None = None

    try:
        with httpx.Client(timeout=_chat_timeout()) as client:
            with client.stream("POST", url, json=stream_payload) as resp:
                if resp.status_code >= 400:
                    body_bytes = resp.read()
                    body = body_bytes.decode("utf-8", errors="replace")[:500]
                    err_msg = f"[{label}] Ollama returned {resp.status_code}: {body}"
                    logger.warning(err_msg)
                    print(err_msg, flush=True)
                    raise LLMError(
                        f"Ollama {resp.status_code} from '{model}'. {_classify_error(model, body)} Body: {body}"
                    )

                print(f"[{label}] stream> ", end="", flush=True)
                thinking_buf: list[str] = []
                first_line_dumped = False
                for line in resp.iter_lines():
                    if not line:
                        continue
                    lines_seen += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"\n[{label}] non-json line: {line[:200]}", flush=True)
                        continue
                    if not first_line_dumped:
                        preview = json.dumps(obj)[:300]
                        logger.info("[%s] first line: %s", label, preview)
                        first_line_dumped = True
                    msg_obj = obj.get("message", {}) or {}
                    chunk = msg_obj.get("content") or ""
                    thinking = msg_obj.get("thinking") or ""
                    if thinking:
                        thinking_buf.append(thinking)
                    if chunk:
                        if first_token_at is None:
                            first_token_at = time.time()
                        print(chunk, end="", flush=True)
                        accumulated.append(chunk)
                    if obj.get("done"):
                        final_obj = obj
                print(flush=True)  # newline after streaming ends
                # Fallback: if the model routed everything into "thinking" and
                # left content empty, try the thinking text. Strip <think>/
                # </think> wrappers that Qwen3 sometimes embeds.
                if not accumulated and thinking_buf:
                    raw_think = "".join(thinking_buf)
                    stripped = re.sub(r"</?think>", "", raw_think).strip()
                    if stripped:
                        logger.warning(
                            "[%s] content empty; falling back to %d chars of 'thinking'.",
                            label, len(stripped),
                        )
                        accumulated.append(stripped)
                if not accumulated and final_obj:
                    logger.warning(
                        "[%s] empty content. done_reason=%s eval_count=%s",
                        label, final_obj.get("done_reason"), final_obj.get("eval_count"),
                    )
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot reach Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise LLMError(f"Ollama request timed out: {e}") from e

    elapsed = time.time() - t0
    content = "".join(accumulated)
    # Preserve any top-level stats Ollama returned (eval_count, total_duration, etc.)
    data = dict(final_obj) if final_obj else {}
    data["message"] = {"role": "assistant", "content": content}

    ttft = f", first-token {first_token_at - t0:.2f}s" if first_token_at else ""
    eval_count = data.get("eval_count")
    tok_info = f", {eval_count} tokens" if eval_count else ""
    done_msg = f"[{label}] <- {elapsed:.2f}s{ttft}{tok_info} | {len(content)} chars"
    logger.info(done_msg)
    print(done_msg, flush=True)
    return data, elapsed


def _coerce_bbox(
    bbox: Any, *, fmt: str, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Convert `bbox` to (x, y, w, h) pixel ints.

    fmt: "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h). Normalized 0-1000 values
    (Qwen-VL quirk) are rescaled to pixels when detected.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        a, b, c, d = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    mx = max(a, b, c, d)
    if mx <= 1000 and mx > max(img_w, img_h) * 1.1:
        a = a / 1000.0 * img_w
        b = b / 1000.0 * img_h
        c = c / 1000.0 * img_w
        d = d / 1000.0 * img_h

    if fmt == "xyxy":
        x, y, w, h = a, b, c - a, d - b
    else:  # xywh
        x, y, w, h = a, b, c, d

    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _extract_regions_list(parsed: Any) -> list[Any]:
    """Find the list of regions. Each element may be a dict (old schema)
    or a list (compact schema: [x1, y1, x2, y2, "text"])."""
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, (dict, list))]
    if not isinstance(parsed, dict):
        return []
    for key in (
        "regions", "chunks", "text_regions", "results", "data", "items",
        "layout", "layout_elements", "elements",
    ):
        v = parsed.get(key)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, (dict, list))]
    if "bbox_2d" in parsed or "bbox" in parsed or "box" in parsed:
        return [parsed]
    return []


def _smart_resize(
    h: int, w: int, *, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 802816
) -> tuple[int, int]:
    """Replicates Qwen2.5-VL's image preprocessor (from transformers
    image_processing_qwen2_vl.smart_resize). Returns (h_bar, w_bar) rounded
    to multiples of `factor` and bounded by min/max pixel budgets. Sending
    an image pre-resized to these dims means the model won't resize further
    — so bboxes come back in the exact space we expect."""
    import math

    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = max(factor, math.ceil(h * beta / factor) * factor)
        w_bar = max(factor, math.ceil(w * beta / factor) * factor)
    return h_bar, w_bar


def _preresize_for_vl(image_path: Path, max_pixels: int) -> tuple[bytes, int, int, int, int]:
    """Resize the page so it matches Qwen-VL's internal smart_resize output.
    Returns (png_bytes, orig_w, orig_h, new_w, new_h)."""
    import io
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        new_h, new_w = _smart_resize(orig_h, orig_w, max_pixels=max_pixels)
        if (new_w, new_h) == (orig_w, orig_h):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue(), orig_w, orig_h, orig_w, orig_h
        resized = im.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue(), orig_w, orig_h, new_w, new_h


def vl_extract_chunks(image_path: Path) -> tuple[list[dict[str, Any]], float]:
    """Call Qwen2.5-VL. Returns (chunks, elapsed_seconds). Each chunk is {text, bbox:[x,y,w,h]}
    in the ORIGINAL image's pixel coordinates."""
    png_bytes, orig_w, orig_h, sent_w, sent_h = _preresize_for_vl(
        image_path, max_pixels=settings.vl_max_pixels
    )
    scale_x = orig_w / sent_w
    scale_y = orig_h / sent_h
    logger.info(
        "[VL] sending %dx%d (from %dx%d); bbox scale x=%.3f y=%.3f",
        sent_w, sent_h, orig_w, orig_h, scale_x, scale_y,
    )

    b64 = base64.b64encode(png_bytes).decode("ascii")
    prompt_text = _vl_prompt()

    if settings.vl_backend == "llamacpp":
        # OpenAI-compatible payload.
        payload: dict[str, Any] = {
            "model": settings.vl_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        if settings.vl_num_predict > 0:
            payload["max_tokens"] = settings.vl_num_predict
        data, elapsed = _post_chat_openai(payload, label="VL", base_url=settings.llamacpp_host)
    else:
        options: dict[str, Any] = {
            "temperature": 0.0,
            # Discourage degenerate repetition loops (e.g. the same region
            # emitted over and over until num_predict is exhausted).
            "repeat_penalty": 1.1,
        }
        if settings.vl_num_ctx > 0:
            options["num_ctx"] = settings.vl_num_ctx
        if settings.vl_num_predict > 0:
            options["num_predict"] = settings.vl_num_predict
        payload = {
            "model": settings.vl_model,
            "messages": [
                {"role": "user", "content": prompt_text, "images": [b64]},
            ],
            "stream": False,
            "options": options,
        }
        if settings.vl_format_json:
            payload["format"] = "json"
        if settings.vl_disable_think:
            payload["think"] = False
        data, elapsed = _post_chat(payload, label="VL")

    content = data.get("message", {}).get("content", "")
    if not content.strip():
        # Empty content most often means the model spent its whole budget in
        # "thinking" mode. Surface a targeted error so the UI shows it.
        raise LLMError(
            f"Model '{settings.vl_model}' returned 0 content chars (likely stayed in "
            "reasoning mode). Try: (1) keep VL_DISABLE_THINK=true, (2) set "
            "VL_FORMAT_JSON=false, (3) raise VL_NUM_PREDICT (e.g. 4096) so reasoning "
            "doesn't exhaust the budget, or (4) try a different VL model "
            "(qwen2.5vl:7b is a known-good choice)."
        )
    try:
        parsed = _safe_json_loads(content)
    except json.JSONDecodeError as e:
        logger.warning("[VL] JSON parse failed: %s | raw: %s", e, _preview(content, 2000))
        return [], elapsed

    # bbox_2d / bbox / box are interpreted in the coord space of the image we SENT.
    img_w, img_h = sent_w, sent_h

    # In dots.ocr mode, the "bbox" field is xyxy (not xywh like the qwen prompt).
    bbox_fmt_default = "xyxy" if settings.vl_prompt_mode == "dotsocr" else "xywh"

    regions = _extract_regions_list(parsed)
    out: list[dict[str, Any]] = []
    for r in regions:
        # Compact array form: [x1, y1, x2, y2, "text"]
        if isinstance(r, list):
            if len(r) < 5:
                continue
            bbox, fmt = r[:4], "xyxy"
            text = str(r[4]).strip()
        elif isinstance(r, dict):
            # dots.ocr uses "category"; skip non-text layout elements.
            category = str(r.get("category", "")).strip()
            if category in _DOTSOCR_SKIP_CATEGORIES:
                continue
            text = str(
                r.get("t")
                or r.get("text_content")
                or r.get("text")
                or r.get("content")
                or ""
            ).strip()
            if "b" in r and isinstance(r["b"], list):
                bbox, fmt = r["b"], "xyxy"
            elif "bbox_2d" in r:
                bbox, fmt = r["bbox_2d"], "xyxy"
            elif "bbox" in r:
                bbox, fmt = r["bbox"], bbox_fmt_default
            elif "box" in r:
                bbox, fmt = r["box"], bbox_fmt_default
            else:
                continue
        else:
            continue
        if not text:
            continue
        if settings.vl_filter_money and _looks_like_money(text):
            continue
        coerced = _coerce_bbox(bbox, fmt=fmt, img_w=img_w, img_h=img_h)
        if coerced is None:
            continue
        sx, sy, sw, sh = coerced
        # Scale from sent-image coords back to original-image coords.
        ox = int(round(sx * scale_x))
        oy = int(round(sy * scale_y))
        ow = max(1, int(round(sw * scale_x)))
        oh = max(1, int(round(sh * scale_y)))
        # Clip to original image bounds.
        ox = max(0, min(ox, orig_w))
        oy = max(0, min(oy, orig_h))
        ow = max(1, min(ow, orig_w - ox))
        oh = max(1, min(oh, orig_h - oy))
        out.append({"text": text, "bbox": [ox, oy, ow, oh]})

    if not out:
        logger.warning(
            "[VL] parsed 0 regions from response. sent=%dx%d orig=%dx%d | raw: %s",
            sent_w, sent_h, orig_w, orig_h, _preview(content, 2000),
        )
    else:
        logger.info(
            "[VL] parsed %d chunks (sent %dx%d, original %dx%d)",
            len(out), sent_w, sent_h, orig_w, orig_h,
        )
    return out, elapsed


def pii_tag_text(text: str) -> tuple[list[dict[str, str]], float]:
    """Call Qwen3. Returns (entities, elapsed_seconds)."""
    if not text.strip():
        return [], 0.0
    options: dict[str, Any] = {"temperature": 0.0}
    if settings.pii_num_ctx > 0:
        options["num_ctx"] = settings.pii_num_ctx
    if settings.pii_num_predict > 0:
        options["num_predict"] = settings.pii_num_predict
    payload = {
        "model": settings.pii_model,
        "messages": [
            {"role": "user", "content": PII_PROMPT_TEMPLATE.format(text=text)},
        ],
        "stream": False,
        "options": options,
    }
    if settings.pii_format_json:
        payload["format"] = "json"
    if settings.pii_disable_think:
        payload["think"] = False
    data, elapsed = _post_chat(payload, label="PII")
    content = data.get("message", {}).get("content", "")
    try:
        parsed = _safe_json_loads(content)
    except json.JSONDecodeError as e:
        logger.warning("[PII] JSON parse failed: %s", e)
        return [], elapsed
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
    out: list[dict[str, str]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type", "")).strip().upper()
        etext = str(e.get("text", "")).strip()
        if not etype or not etext or etype not in PII_TYPES:
            continue
        out.append({"type": etype, "text": etext})
    logger.info("[PII] parsed %d entities", len(out))
    return out, elapsed
