# Thai Government Document OCR Benchmark Pipeline (`train-ocr`)

ระบบรวบรวมเอกสารราชการไทย (PDFs, รูปภาพเอกสาร, หนังสือราชการตามระเบียบงานสารบรรณ 6 ชนิด), คัดกรองคุณภาพการสแกนและตรวจสอบข้อมูลส่วนบุคคล (PII), สร้างชุดข้อมูล Benchmark สำหรับประเมินผล Thai OCR และเครื่องมือติดป้ายกำกับ Ground Truth

---

## 🛠️ โครงสร้างบริการ (Docker Compose Services)

ในไฟล์ [compose.yaml](file:///c:/Users/narawit.s/dev/train_ocr/compose.yaml) ได้จัดสรรสถาปัตยกรรมแบบ Decoupled Services เพื่อแยก dependencies ที่หนัก (เช่น Chrome Headless สำหรับ crawler หรือ RapidOCR สำหรับการกรอง) ออกจากกัน:

| Service Name | Profile | Port / Endpoint | หน้าที่และการทำงาน |
|---|---|---|---|
| **`ocr-screener`** | *(default)* | `8001:8001`<br>`/health`, `/screen` | ไมโครเซอร์วิส FastAPI + RapidOCR สำหรับตรวจวัดคุณภาพภาพเอกสาร (ประเมินกล่องข้อความ, ทิศทาง, อักขระภาษาไทย, คุณภาพการสแกน) เพื่อคัดแยกเอกสารที่มีข้อความจริงออกจากภาพขยะ |
| **`crawler-portal`** | `portal` | - | บิลด์สำหรับงานรวบรวมเอกสารจาก Government Open Data Portal (`data.go.th`) และกรมส่งเสริมการปกครองท้องถิ่น (`dla.go.th`, `sll.dla.go.th`) |
| **`crawler-google`** | `google` | `7900:7900`<br>(noVNC Web UI) | ติดตั้ง Google Chrome + Xvfb + noVNC สำหรับรัน AutoCrawler ดึงรูปภาพตัวอย่างหนังสือราชการและเอกสารราชการผ่าน Google Images พร้อมดูหน้าจอผ่านเว็บเบราว์เซอร์ได้ที่พอร์ต 7900 |
| **`dataset-publisher`** | `publisher` | - | บริการประมวลผล PDF เป็นภาพความละเอียดสูง (200 / 300 DPI), แปลงเป็น Hugging Face Dataset format (Parquet/Arrow) และอัปโหลดขึ้น Hugging Face Hub |

```bash
# รันเฉพาะ screener service
docker compose up -d ocr-screener

# รัน crawler portal
docker compose --profile portal run --rm crawler-portal python -m train_ocr.cli collect-dla

# รัน google image crawler (ดู UI ได้ที่ http://localhost:7900)
docker compose --profile google up crawler-google
```

---

## 📦 โครงสร้างโปรเจกต์ & สิ่งที่มีในระบบ (What's Inside)

```text
train_ocr/
├── src/train_ocr/                     # Core Python Package
│   ├── cli.py                         # Unified CLI Dispatcher (`train-ocr <cmd>`)
│   ├── collectors/                    # โมดูลรวบรวมข้อมูลจากแหล่งต่างๆ
│   │   ├── datagoth.py                # รวบรวม PDF จาก data.go.th CKAN API
│   │   ├── dla.py                     # รวบรวมหนังสือราชการ 6 ชนิดสารบรรณ (sampling mode)
│   │   ├── dla_bulk/                  # ตัวเก็บเอกสาร DLA ขนาดใหญ่ (ครอบคลุม 12 พอร์ทัล)
│   │   │   ├── catalog.py             # SQLite WAL catalog & state checkpointing
│   │   │   ├── portals.py             # ตัวดึงข้อมูล 12 พอร์ทัล (Struts POST, DWR e-Book, SLL tree)
│   │   │   └── downloader.py          # Concurrent downloader (PDF+รูปภาพ, deduplication, budget cap)
│   │   └── google_images.py           # ตัวเชื่อมต่อ AutoCrawler ดึงภาพจาก Google Search
│   ├── screener/                      # โมดูลคัดกรองคุณภาพภาพและสแกน PDF
│   │   ├── api.py                     # FastAPI REST API (`/screen`)
│   │   ├── engine.py                  # ตัวคำนวณ RapidOCR metrics และภาษาไทย
│   │   ├── filter_images.py           # สคริปต์คัดแยกรูปภาพเข้า kept/ หรือ rejected/
│   │   └── pdf_scanner.py             # ตัววิเคราะห์หน้า PDF แบบแยกแยะ scanned vs digital
│   ├── common/                        # โมดูลช่วยเหลือส่วนกลาง
│   │   ├── settings.py                # ค่าคอนฟิกูเรชัน พาธ และงบประมาณพื้นที่
│   │   ├── pdf_io.py                  # ฟังก์ชันจัดการ PDF ผ่าน pypdfium2 และคำนวณ image coverage
│   │   ├── saraban_rules.py           # กฎตรวจจับประเภทเอกสารสารบรรณ 6 ชนิด และคัดกรอง PII
│   │   └── manifest.py                # ตัวจัดการ JSONL Manifest
│   ├── dataset/
│   │   └── builder.py                 # ตัวแปลงภาพและสร้าง Hugging Face Dataset (200/300 DPI)
│   └── publisher/
│       └── hf_hub.py                  # ตัวเผยแพร่ชุดข้อมูลขึ้น Hugging Face Hub
│
├── Root Backward-Compatible Shims     # สคริปต์เดิมที่คงไว้เพื่อ backward compatibility
│   ├── collect_dla_bulk.py            # สคริปต์รัน DLA Bulk Harvester โดยตรง
│   ├── collect_dla.py                 # ดึงหนังสือราชการ DLA 6 ชนิดสารบรรณ
│   ├── collect_pdfs.py                # ดึง PDF ทั่วไปจาก data.go.th
│   ├── collect_saraban_extra.py       # ดึงเอกสารสารบรรณเพิ่มเติมจาก data.go.th
│   ├── eda_opendata.py                # สำรวจ metadata จาก data.go.th CKAN API
│   ├── scan_pages.py                  # สแกนจัดประเภทหน้า PDF ทั่วไป
│   ├── scan_saraban_pages.py          # สแกนจัดประเภทหน้า PDF หนังสือราชการสารบรรณ
│   ├── prepare_ocr_dataset.py         # ประกอบ dataset ทั่วไป
│   ├── prepare_saraban_dataset.py     # ประกอบ dataset สารบรรณ 300 DPI
│   ├── push_to_hf.py                  # อัปโหลดชุดข้อมูลขึ้น Hugging Face
│   ├── config.py                      # Re-export ค่าคอนฟิกและ credential loader
│   └── saraban_utils.py               # Re-export ฟังก์ชันตรวจสอบ PII และ regex สารบรรณ
│
├── Evaluation & Ground Truth Tools    # ชุดเครื่องมือประเมินผลและทำ Ground Truth
│   ├── select_golden_set.py           # คัดเลือก Golden Benchmark Set (120 หน้า) แบบกระจายทุกชนิด
│   ├── draft_gt_dual.py               # สร้าง Ground Truth ร่างแบบคู่ขนาน (Gemini Pro + KKU/iApp)
│   ├── review_ui.py                   # เว็บ UI สำหรับตรวจทานและอนุมัติ Ground Truth
│   ├── eval_ocr_benchmark.py          # คำนวณค่า CER, WER, Edit Distance เปรียบเทียบโมเดล OCR
│   └── sync_gt_to_hf.py               # ซิงค์ชุดข้อมูล Ground Truth ที่ตรวจแล้วขึ้น HF
│
├── docker/                            # Dockerfiles & scripts ประจำแต่ละ service
├── tests/                             # Unit tests (pytest) ครอบคลุมระบบ CLI, Core, DLA Bulk, Screener
├── pyproject.toml / requirements.txt  # Project dependencies และ setup script
└── compose.yaml                       # Docker Compose orchestration
```

---

## 🚀 คำสั่งใช้งานหลัก (Unified CLI)

ระบบมี CLI หลักผ่านโมดูล `train_ocr.cli`:

```powershell
# ดูคำสั่งทั้งหมด
.venv\Scripts\python.exe -m train_ocr.cli --help
```

### 1. การเก็บเอกสารจาก กรมส่งเสริมการปกครองท้องถิ่น (DLA Bulk Harvester)
เก็บเอกสารขนาดใหญ่จาก `www.dla.go.th` และ `sll.dla.go.th` ครอบคลุมทั้ง 12 หมวด (หนังสือราชการ, ข่าวประชาสัมพันธ์จัดซื้อจัดจ้าง, ตอบข้อหารือกฎหมาย, e-Book, SLL ฯลฯ) รองรับการหยุดแล้วรันต่อ (Resumable ผ่าน SQLite):

```powershell
# รันเก็บเอกสารแบบ Bulk (ค่าเริ่มต้น: โหลด PDF + รูปภาพ, เก็บครบทุกหมวด, จำกัดเพดาน 1 GB)
.venv\Scripts\python.exe collect_dla_bulk.py

# หรือกำหนดพารามิเตอร์เพิ่มเติม
.venv\Scripts\python.exe collect_dla_bulk.py --workers 4 --max-total-mb 2048 --sources official_doc,legal_item,ebook,sll

# ตรวจสอบสถิติจำนวนเอกสารและไฟล์ที่ดาวน์โหลดแล้วใน Catalog
.venv\Scripts\python.exe collect_dla_bulk.py --phase stats
```

### 2. กระบวนการ Saraban Benchmark Pipeline (6 ชนิดหนังสือราชการ)
1. **รวบรวมเอกสาร**:
   ```powershell
   .venv\Scripts\python.exe -m train_ocr.cli collect-dla --per-type 45
   .venv\Scripts\python.exe -m train_ocr.cli collect-saraban-extra
   ```
2. **สแกนและตรวจสอบหน้าเอกสาร (Scanned vs Digital / ตรวจ PII)**:
   ```powershell
   .venv\Scripts\python.exe -m train_ocr.cli scan-saraban
   ```
3. **ประกอบชุดข้อมูล Benchmark 300 DPI**:
   ```powershell
   .venv\Scripts\python.exe -m train_ocr.cli build-saraban
   ```
4. **อัปโหลดสู่ Hugging Face Hub**:
   ```powershell
   .venv\Scripts\python.exe -m train_ocr.cli push-hf
   ```

### 3. ชุดเครื่องมือ Ground Truth & Benchmark Evaluation
- **คัดเลือกหน้าเอกสาร Golden Set (120 หน้า)**:
  ```powershell
  .venv\Scripts\python.exe select_golden_set.py
  ```
- **สร้างคำอ่านร่าง (Dual-model drafting)**:
  ```powershell
  .venv\Scripts\python.exe draft_gt_dual.py
  ```
- **เปิด Review UI เพื่อตรวจและแก้คำอ่าน**:
  ```powershell
  .venv\Scripts\python.exe review_ui.py
  ```
- **ประเมินผลคะแนนความแม่นยำ OCR (CER / WER)**:
  ```powershell
  .venv\Scripts\python.exe eval_ocr_benchmark.py
  ```

### 4. กระบวนการเก็บภาพเอกสารจาก Google Images (`google_docs`)
1. **ดึงรูปภาพความละเอียดเต็ม** (รองรับ `--offset` เพื่อข้ามช่วงที่เคยดึงแล้วโดยไม่ทับไฟล์เดิม):
   ```powershell
   .venv\Scripts\train-ocr.exe collect-google --keywords-file data/google_images/keywords.txt --download-dir data/google_images/raw --profile-dir data/google_images/chrome-profile --full --no-no-gui --limit 40
   # ดึงต่อช่วง 40-100:
   .venv\Scripts\train-ocr.exe collect-google --keywords-file data/google_images/keywords.txt --download-dir data/google_images/raw --profile-dir data/google_images/chrome-profile --full --no-no-gui --offset 40 --limit 100
   ```
2. **คัดกรองคุณภาพด้วย RapidOCR + SHA256 deduplication**:
   ```powershell
   .venv\Scripts\train-ocr.exe filter-images --input-dir data/google_images/raw --output-dir data/google_images/filtered --min-dim 800
   ```
3. **Preprocess & Build HF Dataset** (แก้ EXIF orientation, แปลง RGB PNG, ตรวจ PII และคง GT เดิมจาก `data/google_images/gt/gt_annotations.jsonl`):
   ```powershell
   .venv\Scripts\train-ocr.exe build-google
   .venv\Scripts\train-ocr.exe push-hf --config-name google_docs
   ```

---

## 🧪 การทดสอบระบบ (Tests)

รัน unit test ทั้งหมดด้วย `pytest`:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_dla_bulk.py tests/test_cli_and_shims.py tests/test_common.py tests/test_screener_api.py
```
