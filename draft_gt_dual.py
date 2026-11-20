"""Dual-model GT drafter using KKU GenAI OpenAI-compatible API (2 cross-family VLMs)."""

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image
from rapidfuzz import fuzz

from config import (
    GOLDEN_DRAFTS_PATH,
    GOLDEN_MANIFEST_PATH,
    KKU_BASE_URL,
    KKU_MODEL_A,
    KKU_MODEL_B,
    load_kku_api_key,
)

OCR_TRANSCRIPTION_PROMPT = """คุณคือผู้เชี่ยวชาญการถอดความเอกสารราชการไทย (Thai Saraban OCR Ground Truth Annotator)
จงถอดความเอกสารในภาพนี้อย่างละเอียดและถูกต้อง 100% ตามกฎต่อไปนี้:

1. **ตัวเลขไทยและเลขอารบิก:** เก็บตัวเลขตามที่ปรากฏจริงบนกระดาษเป๊ะๆ (ถ้าพิมพ์เลขไทย ๑–๙ ให้ถอดเป็นเลขไทย, ถ้าพิมพ์เลขอารบิก 0–9 ให้ถอดเป็นเลขอารบิก ห้ามแปลงข้ามกัน)
2. **การจัดรูปแบบ:** ถอดข้อความเป็น Markdown ตามลำดับการอ่าน (Reading Order)
   - ตารางธรรมดาใช้ Markdown Table (`| ... |`)
   - ตารางซับซ้อนที่มีการผสานเซลล์ (merged cells) ให้ใช้ HTML `<table>` พร้อม `rowspan`/`colspan`
3. **ตรายาง ลายเซ็น และลายมือเกษียน:**
   - เก็บข้อความพิมพ์หลักทั้งหมด + ข้อความในตรายางที่อ่านชัด (เช่น "ด่วนที่สุด", เลขรับหนังสือ, ตราประทับหน่วยงาน)
   - สำหรับลายเซ็นที่อ่านเป็นตัวอักษรไม่ได้ ให้ใส่แท็ก `[SIGNATURE]` แทน
   - ไม่ต้องถอดความลายมือเขียนเกษียนหนังสือสั้นๆ ข้างขอบกระดาษ
4. **KIE Header Fields (`kie`):** ดึงฟิลด์หัวหนังสือจากเอกสารในภาพ (ถ้าไม่ปรากฏในหน้านี้ให้ใส่ `""`):
   - `doc_no`: เลขที่หนังสือ (เช่น "ที่ มท ๐๘๐๘.๒/ว ๑๒๓๔" หรือ "มท 0808.2/ว 1234")
   - `doc_date`: วันที่ของหนังสือ
   - `subject`: ชื่อเรื่อง

ตอบกลับเป็น JSON เท่านั้น ในรูปแบบ:
{
  "markdown": "...",
  "kie": {
    "doc_no": "...",
    "doc_date": "...",
    "subject": "..."
  }
}"""


def _parse_json_response(raw_text: str) -> tuple[str, dict]:
    """Parse JSON (stripping optional markdown code fences) into (markdown, kie)."""
    text = raw_text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```$", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        md = str(data.get("markdown") or "").strip()
        kie = data.get("kie") or {}
        return md, {
            "doc_no": str(kie.get("doc_no") or "").strip(),
            "doc_date": str(kie.get("doc_date") or "").strip(),
            "subject": str(kie.get("subject") or "").strip(),
        }
    except json.JSONDecodeError:
        return text, {"doc_no": "", "doc_date": "", "subject": ""}


def _encode_image_for_api(image_path: Path, max_width: int = 1600) -> str:
    """Resize if needed and encode as JPEG (quality=90) to stay under API payload limits."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if w > max_width:
            rgb = rgb.resize((max_width, max(1, int(h * max_width / w))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def call_vlm_openai_compat(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    timeout: int = 180,
) -> tuple[str, dict]:
    """Call an OpenAI-compatible /chat/completions endpoint with compressed JPEG image."""
    b64_img = _encode_image_for_api(image_path)
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_TRANSCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    },
                ],
            }
        ],
    }
    for attempt in range(2):
        try:
            resp = session.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_json_response(content)
        except requests.Timeout:
            if attempt == 0:
                time.sleep(2.0)
                continue
            raise


def compute_agreement(text_a: str, text_b: str) -> float:
    """Return normalized character-level similarity (0.0 - 1.0) between two drafts."""
    a = " ".join((text_a or "").split())
    b = " ".join((text_b or "").split())
    if not a or not b:
        return 0.0
    return round(fuzz.ratio(a, b) / 100.0, 4)


def run_drafting(
    base_url: str = KKU_BASE_URL,
    model_a: str = KKU_MODEL_A,
    model_b: str = KKU_MODEL_B,
    only: str = "both",
    limit: int = 0,
    delay: float = 0.5,
) -> None:
    if not GOLDEN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Golden manifest not found: {GOLDEN_MANIFEST_PATH}. Run select_golden_set.py first.")

    api_key = load_kku_api_key()
    manifest = [
        json.loads(line)
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict] = {}
    if GOLDEN_DRAFTS_PATH.exists():
        for line in GOLDEN_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["sample_id"]] = rec

    session = requests.Session()
    processed = 0

    for item in manifest:
        sid = item["sample_id"]
        img_path = Path(item["image_path"])
        if not img_path.exists():
            print(f"[warn] Missing image for {sid}: {img_path}")
            continue

        rec = existing.get(
            sid,
            {
                "sample_id": sid,
                "model_a_name": model_a,
                "model_a_markdown": "",
                "model_a_kie": {},
                "model_b_name": model_b,
                "model_b_markdown": "",
                "model_b_kie": {},
                "agreement_score": 0.0,
            },
        )

        changed = False
        if only in {"both", "a"} and not rec.get("model_a_markdown"):
            try:
                md, kie = call_vlm_openai_compat(session, base_url, api_key, model_a, img_path)
                rec["model_a_name"] = model_a
                rec["model_a_markdown"] = md
                rec["model_a_kie"] = kie
                changed = True
            except Exception as exc:
                print(f"  [err] {model_a} failed for {sid}: {exc}")

        if only in {"both", "b"} and not rec.get("model_b_markdown"):
            try:
                md, kie = call_vlm_openai_compat(session, base_url, api_key, model_b, img_path)
                rec["model_b_name"] = model_b
                rec["model_b_markdown"] = md
                rec["model_b_kie"] = kie
                changed = True
            except Exception as exc:
                print(f"  [err] {model_b} failed for {sid}: {exc}")

        if changed:
            rec["agreement_score"] = compute_agreement(
                rec.get("model_a_markdown", ""), rec.get("model_b_markdown", "")
            )
            existing[sid] = rec
            GOLDEN_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with GOLDEN_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                for r in existing.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            processed += 1
            print(
                f"[{processed}] {sid} ({item['saraban_type']}/{item['page_type']}) "
                f"agreement={rec['agreement_score']:.3f}"
            )
            time.sleep(delay)

    print(f"\nDual drafting complete: {len(existing)}/{len(manifest)} samples stored in {GOLDEN_DRAFTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dual-model GT drafts via OpenAI-compatible VLM API.")
    parser.add_argument("--base-url", type=str, default=KKU_BASE_URL)
    parser.add_argument("--model-a", type=str, default=KKU_MODEL_A)
    parser.add_argument("--model-b", type=str, default=KKU_MODEL_B)
    parser.add_argument("--only", choices=["both", "a", "b"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples (0 = all).")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    run_drafting(
        base_url=args.base_url,
        model_a=args.model_a,
        model_b=args.model_b,
        only=args.only,
        limit=args.limit,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
