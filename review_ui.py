"""Local FastAPI Web UI for Human-in-the-Loop Ground Truth Verification (localhost:8765)."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import GOLDEN_DRAFTS_PATH, GOLDEN_MANIFEST_PATH, GT_ANNOTATIONS_PATH

app = FastAPI(title="Thai Saraban OCR GT Review UI")

QUALITY_TAG_OPTIONS = [
    "has_red_stamp",
    "has_blue_ink",
    "low_contrast",
    "skewed",
    "handwriting_note",
    "has_table",
    "broken_font",
]


class SaveAnnotationRequest(BaseModel):
    sample_id: str
    ground_truth_text: str
    gt_status: str = "human_verified"
    kie: dict[str, str] = {}
    quality_tags: list[str] = []


def _load_jsonl_map(path: Path, key: str = "sample_id") -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec[key]] = rec
    return out


def _write_jsonl_map(path: Path, data: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in data.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.get("/api/items")
def get_items() -> dict:
    """Return merged Golden Set manifest + dual-model drafts + saved GT annotations."""
    if not GOLDEN_MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="golden_manifest.jsonl not found")

    manifest = [
        json.loads(line)
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    drafts = _load_jsonl_map(GOLDEN_DRAFTS_PATH)
    annotations = _load_jsonl_map(GT_ANNOTATIONS_PATH)

    items: list[dict] = []
    for m in manifest:
        sid = m["sample_id"]
        d = drafts.get(sid, {})
        a = annotations.get(sid, {})
        g_kie = d.get("model_a_kie") or d.get("gemini_kie") or {}
        md_a = d.get("model_a_markdown") or d.get("gemini_markdown") or ""
        md_b = d.get("model_b_markdown") or d.get("iapp_markdown") or ""
        default_kie = {
            "doc_no": g_kie.get("doc_no") or m.get("meta_doc_no") or "",
            "doc_date": g_kie.get("doc_date") or m.get("meta_doc_date") or "",
            "subject": g_kie.get("subject") or m.get("meta_subject") or "",
        }
        items.append(
            {
                **m,
                "model_a_name": d.get("model_a_name", "Model A"),
                "model_b_name": d.get("model_b_name", "Model B"),
                "gemini_markdown": md_a,
                "iapp_markdown": md_b,
                "agreement_score": d.get("agreement_score", 0.0),
                "ground_truth_text": a.get("ground_truth_text")
                or md_a
                or md_b
                or "",
                "gt_status": a.get("gt_status", "unverified"),
                "kie": a.get("kie") or default_kie,
                "saved_quality_tags": a.get("quality_tags")
                if "quality_tags" in a
                else m.get("quality_tags", []),
            }
        )

    verified = sum(1 for x in items if x["gt_status"] == "human_verified")
    return {"total": len(items), "verified": verified, "items": items, "quality_tag_options": QUALITY_TAG_OPTIONS}


@app.get("/api/image/{sample_id}")
def get_image(sample_id: str) -> FileResponse:
    """Serve the 300 DPI PNG image for a given sample_id."""
    manifest = _load_jsonl_map(GOLDEN_MANIFEST_PATH)
    item = manifest.get(sample_id)
    if not item:
        raise HTTPException(status_code=404, detail="Sample not found")
    img_path = Path(item["image_path"])
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(str(img_path), media_type="image/png")


@app.post("/api/save")
def save_annotation(req: SaveAnnotationRequest) -> dict:
    """Persist a reviewed Ground Truth annotation to gt_annotations.jsonl."""
    annotations = _load_jsonl_map(GT_ANNOTATIONS_PATH)
    annotations[req.sample_id] = {
        "sample_id": req.sample_id,
        "ground_truth_text": req.ground_truth_text,
        "gt_status": req.gt_status,
        "kie": {
            "doc_no": (req.kie.get("doc_no") or "").strip(),
            "doc_date": (req.kie.get("doc_date") or "").strip(),
            "subject": (req.kie.get("subject") or "").strip(),
        },
        "quality_tags": req.quality_tags,
    }
    _write_jsonl_map(GT_ANNOTATIONS_PATH, annotations)
    verified = sum(1 for v in annotations.values() if v.get("gt_status") == "human_verified")
    return {"ok": True, "sample_id": req.sample_id, "verified": verified}


HTML_PAGE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8" />
<title>Thai Saraban OCR GT Review UI</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
  header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #1e293b; border-bottom: 1px solid #334155; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; color: #38bdf8; }
  .pill { background: #334155; padding: 3px 8px; border-radius: 6px; font-size: 12px; }
  .pill.ok { background: #14532d; color: #86efac; }
  .pill.warn { background: #713f12; color: #fde047; }
  select, button { background: #334155; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
  button.primary { background: #0284c7; border-color: #0369a1; font-weight: 600; }
  button.success { background: #16a34a; border-color: #15803d; font-weight: 600; }
  button.danger { background: #b91c1c; border-color: #991b1b; }
  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; overflow: hidden; }
  .left { position: relative; overflow: auto; background: #020617; border-right: 1px solid #334155; display: flex; flex-direction: column; }
  .img-toolbar { position: sticky; top: 0; z-index: 5; background: rgba(15,23,42,0.92); padding: 8px 12px; display: flex; gap: 8px; align-items: center; border-bottom: 1px solid #1e293b; }
  .img-wrap { flex: 1; overflow: auto; padding: 12px; text-align: center; }
  .img-wrap img { transform-origin: top left; border: 1px solid #334155; background: #fff; max-width: none; }
  .right { overflow: auto; padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
  .panel { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; }
  .panel h3 { margin: 0 0 8px 0; font-size: 13px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; }
  .diff-box { max-height: 180px; overflow: auto; font-family: "Sarabun", monospace; font-size: 13px; line-height: 1.5; background: #0f172a; padding: 8px; border-radius: 6px; white-space: pre-wrap; }
  .diff-eq { color: #cbd5e1; }
  .diff-del { background: rgba(239,68,68,0.28); color: #fca5a5; text-decoration: line-through; }
  .diff-ins { background: rgba(34,197,94,0.28); color: #86efac; }
  textarea { width: 100%; min-height: 260px; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 10px; font-family: "Sarabun", monospace; font-size: 14px; line-height: 1.55; resize: vertical; }
  .kie-grid { display: grid; grid-template-columns: 110px 1fr; gap: 6px 10px; align-items: center; font-size: 13px; }
  .kie-grid input { width: 100%; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 5px 8px; font-size: 13px; }
  .tags { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; }
  .tags label { display: flex; align-items: center; gap: 4px; background: #0f172a; padding: 4px 8px; border-radius: 6px; border: 1px solid #334155; cursor: pointer; }
  .actions { display: flex; gap: 10px; justify-content: flex-end; align-items: center; }
</style>
</head>
<body>
<header>
  <h1>Saraban OCR GT Review</h1>
  <span id="progress" class="pill">0 / 0 verified</span>
  <select id="filterMode" onchange="applyFilter()">
    <option value="all">All</option>
    <option value="unverified">Unverified Only</option>
    <option value="low_agree">Lowest Agreement First</option>
  </select>
  <button onclick="prevItem()">&larr; Prev</button>
  <span id="idxLabel" class="pill">0 / 0</span>
  <button onclick="nextItem()">Next &rarr;</button>
  <span id="metaBadges" style="display:flex;gap:6px;flex-wrap:wrap;"></span>
</header>
<main>
  <div class="left">
    <div class="img-toolbar">
      <button onclick="setZoom(zoom + 0.15)">Zoom +</button>
      <button onclick="setZoom(Math.max(0.2, zoom - 0.15))">Zoom -</button>
      <button onclick="fitZoom()">Fit Width</button>
      <span id="zoomLabel" class="pill">100%</span>
      <span id="docInfo" style="font-size:12px;color:#94a3b8;"></span>
    </div>
    <div class="img-wrap" id="imgWrap">
      <img id="docImg" src="" alt="Document page" />
    </div>
  </div>
  <div class="right">
    <div class="panel">
      <h3>
        <span>Dual-Model Diff (Gemini vs iApp OpenThai2.1)</span>
        <span style="display:flex;gap:6px;">
          <button onclick="copyDraft('gemini')">Copy Gemini</button>
          <button onclick="copyDraft('iapp')">Copy iApp</button>
          <button onclick="copyDraft('embedded')">Copy Embedded</button>
        </span>
      </h3>
      <div id="diffBox" class="diff-box"></div>
    </div>

    <div class="panel">
      <h3><span>Ground Truth Markdown (Ctrl+Enter = Verify &amp; Next)</span><span id="statusBadge" class="pill"></span></h3>
      <textarea id="gtEditor"></textarea>
    </div>

    <div class="panel">
      <h3>KIE Header Fields</h3>
      <div class="kie-grid">
        <label>doc_no</label><input id="kieDocNo" type="text" />
        <label>doc_date</label><input id="kieDocDate" type="text" />
        <label>subject</label><input id="kieSubject" type="text" />
      </div>
    </div>

    <div class="panel">
      <h3>Quality Tags</h3>
      <div id="tagBox" class="tags"></div>
    </div>

    <div class="actions">
      <button class="danger" onclick="saveCurrent('flagged', true)">Flag / Skip</button>
      <button class="primary" onclick="saveCurrent('ocr_draft', false)">Save Draft</button>
      <button class="success" onclick="saveCurrent('human_verified', true)">Verify &amp; Next (Ctrl+Enter)</button>
    </div>
  </div>
</main>
<script>
let allItems = [], filtered = [], cur = 0, zoom = 0.45, tagOptions = [];

function wordDiffHtml(a, b) {
  if (!a && !b) return '<span class="diff-eq">(No dual-model drafts yet — run draft_gt_dual.py)</span>';
  if (!b) return '<span class="diff-eq">' + esc(a) + '</span>';
  if (!a) return '<span class="diff-eq">' + esc(b) + '</span>';
  const wa = a.split(/(\\s+)/), wb = b.split(/(\\s+)/);
  // Simple bounded LCS diff on first 400 tokens for responsive UI
  const n = Math.min(wa.length, 400), m = Math.min(wb.length, 400);
  const dp = Array.from({length: n+1}, () => new Int32Array(m+1));
  for (let i = n-1; i >= 0; i--) {
    for (let j = m-1; j >= 0; j--) {
      dp[i][j] = wa[i] === wb[j] ? 1 + dp[i+1][j+1] : Math.max(dp[i+1][j], dp[i][j+1]);
    }
  }
  let i = 0, j = 0, html = '';
  while (i < n && j < m) {
    if (wa[i] === wb[j]) { html += '<span class="diff-eq">' + esc(wa[i]) + '</span>'; i++; j++; }
    else if (dp[i+1][j] >= dp[i][j+1]) { html += '<span class="diff-del">' + esc(wa[i]) + '</span>'; i++; }
    else { html += '<span class="diff-ins">' + esc(wb[j]) + '</span>'; j++; }
  }
  while (i < n) html += '<span class="diff-del">' + esc(wa[i++]) + '</span>';
  while (j < m) html += '<span class="diff-ins">' + esc(wb[j++]) + '</span>';
  return html;
}
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function init() {
  const res = await fetch('/api/items');
  const data = await res.json();
  allItems = data.items;
  tagOptions = data.quality_tag_options;
  renderTagBoxes();
  updateProgress();
  applyFilter();
}

function renderTagBoxes() {
  const box = document.getElementById('tagBox');
  box.innerHTML = tagOptions.map(t => `<label><input type="checkbox" value="${t}" /> ${t}</label>`).join('');
}

function updateProgress() {
  const ver = allItems.filter(x => x.gt_status === 'human_verified').length;
  document.getElementById('progress').textContent = `${ver} / ${allItems.length} verified`;
}

function applyFilter() {
  const mode = document.getElementById('filterMode').value;
  if (mode === 'unverified') filtered = allItems.filter(x => x.gt_status !== 'human_verified');
  else if (mode === 'low_agree') filtered = [...allItems].sort((a,b) => a.agreement_score - b.agreement_score);
  else filtered = [...allItems];
  cur = 0;
  renderCurrent();
}

function renderCurrent() {
  if (!filtered.length) {
    document.getElementById('idxLabel').textContent = '0 / 0';
    return;
  }
  const it = filtered[cur];
  document.getElementById('idxLabel').textContent = `${cur + 1} / ${filtered.length}`;
  document.getElementById('docImg').src = `/api/image/${it.sample_id}`;
  document.getElementById('docInfo').textContent = `${it.organization} | p.${it.page_num}/${it.total_pages}`;
  document.getElementById('metaBadges').innerHTML = [
    `<span class="pill">${it.saraban_type}</span>`,
    `<span class="pill">${it.page_type}</span>`,
    `<span class="pill">${it.page_role}</span>`,
    `<span class="pill ${it.agreement_score >= 0.85 ? 'ok' : 'warn'}">agree: ${(it.agreement_score*100).toFixed(1)}%</span>`
  ].join('');
  document.getElementById('diffBox').innerHTML = wordDiffHtml(it.gemini_markdown, it.iapp_markdown);
  document.getElementById('gtEditor').value = it.ground_truth_text || '';
  document.getElementById('kieDocNo').value = (it.kie && it.kie.doc_no) || '';
  document.getElementById('kieDocDate').value = (it.kie && it.kie.doc_date) || '';
  document.getElementById('kieSubject').value = (it.kie && it.kie.subject) || '';
  const st = document.getElementById('statusBadge');
  st.textContent = it.gt_status;
  st.className = 'pill ' + (it.gt_status === 'human_verified' ? 'ok' : 'warn');
  const checked = new Set(it.saved_quality_tags || []);
  document.querySelectorAll('#tagBox input').forEach(cb => cb.checked = checked.has(cb.value));
  fitZoom();
}

function setZoom(z) {
  zoom = z;
  const img = document.getElementById('docImg');
  img.style.width = Math.round(img.naturalWidth * zoom) + 'px';
  document.getElementById('zoomLabel').textContent = Math.round(zoom * 100) + '%';
}
function fitZoom() {
  const img = document.getElementById('docImg');
  const wrap = document.getElementById('imgWrap');
  if (img.naturalWidth) setZoom((wrap.clientWidth - 40) / img.naturalWidth);
  else img.onload = () => setZoom((wrap.clientWidth - 40) / img.naturalWidth);
}
function copyDraft(which) {
  const it = filtered[cur];
  if (!it) return;
  const txt = which === 'gemini' ? it.gemini_markdown : which === 'iapp' ? it.iapp_markdown : it.embedded_text_preview;
  document.getElementById('gtEditor').value = txt || '';
}
function prevItem() { if (cur > 0) { cur--; renderCurrent(); } }
function nextItem() { if (cur < filtered.length - 1) { cur++; renderCurrent(); } }

async function saveCurrent(status, advance) {
  const it = filtered[cur];
  if (!it) return;
  const tags = [...document.querySelectorAll('#tagBox input:checked')].map(cb => cb.value);
  const payload = {
    sample_id: it.sample_id,
    ground_truth_text: document.getElementById('gtEditor').value,
    gt_status: status,
    kie: {
      doc_no: document.getElementById('kieDocNo').value,
      doc_date: document.getElementById('kieDocDate').value,
      subject: document.getElementById('kieSubject').value,
    },
    quality_tags: tags,
  };
  await fetch('/api/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  it.ground_truth_text = payload.ground_truth_text;
  it.gt_status = status;
  it.kie = payload.kie;
  it.saved_quality_tags = tags;
  updateProgress();
  if (advance && cur < filtered.length - 1) { cur++; renderCurrent(); }
  else renderCurrent();
}

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); saveCurrent('human_verified', true); }
});
init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
