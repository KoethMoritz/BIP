# BIP – Metadata Extraction from BVG Construction Plans

This project compares how well different approaches **automatically extract metadata from scanned construction plans of the Berlin U-Bahn network (BVG)**. Four fields are extracted from every plan:

| Field | Meaning |
|-------|---------|
| **Title** | Plan title, name of the structure or plan content |
| **Date** | Creation, review or revision date |
| **ID** | Plan, drawing, sheet or document number |
| **Scale** | Drawing scale, e.g. `1:100`, `1:50` |

Three extraction approaches are compared against a manually created ground truth:

1. **Large vision-language model (VLM) via the university API.** A Qwen model on the HTW server, with reasoning ("thinking") enabled.
2. **Small local VLM.** `qwen2.5vl:3b` running locally through Ollama.
3. **Classical OCR.** Tesseract with OpenCV preprocessing and rule-based field parsing.

---

## Repository structure

```
application_api_v1.py    # Extraction with the university API model (Qwen, thinking mode)
application_llm_v1.py    # Extraction with the local model qwen2.5vl:3b (Ollama)
application_ocr_v4.py    # Extraction with Tesseract OCR (current version)
evaluate_results.py      # Evaluation of a results file against the ground truth
old_data/                # Earlier OCR versions v1–v3 (development history)
requirements.txt
```

---

## Workflow

```
Plans (.tif/.png/.jpg/.pdf ...)
        │
        ├─► application_api_v1.py  ─► results_api.txt
        ├─► application_llm_v1.py  ─► results_local_vlm.txt
        └─► application_ocr_v4.py  ─► results_ocr_v4.txt (+ .csv, debug images)
                                          │
Ground truth (.xlsx/.csv) ────────────────┴─► evaluate_results.py ─► eval_output/
```

When started, each extraction script asks for a folder of plans. It processes every supported file (`.tif .tiff .png .jpg .jpeg .webp .bmp .pdf`) and writes one block per file or page in the same format:

```
FILE: plan_001.tif
----------------------------------------
Title: ...
Date: ...
ID: ...
Scale: ...
============================================================
```

---

## The extraction scripts

### `application_api_v1.py`: University API model
- Loads `HTW_API_KEY`, `HTW_BASE_URL` and `MODEL_NAME` from `.env`.
- Renders PDFs page by page at 200 DPI. Every image is scaled down to at most 2048 px and sent to the model as a JPEG.
- The **whole plan** is sent, not a crop, with an English extraction prompt.
- Thinking mode is on (`enable_thinking`, `reasoning_effort="high"`, `temperature=0.6`, `max_tokens=4096`). `<think>…</think>` blocks are removed from the answer.
- Output: `results_api.txt`. The file is opened in append mode, so repeated runs add to it.

### `application_llm_v1.py`: Local small model
- Uses Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`) with `qwen2.5vl:3b`.
- **Crops the title block.** Following DIN EN ISO 7200, the title block sits in the bottom-right corner, so only the right 45 % and bottom 35 % of the plan is sent. This lets the small 3B model read the text at a higher effective resolution.
- Crops are scaled to at most 1536 px. The prompt is in German and BVG-specific. `temperature=0.1` gives near-deterministic output.
- Output: `results_local_vlm.txt` (append mode).

### `application_ocr_v4.py`: Classical OCR pipeline
A fully rule-based approach that uses no language model:
1. **Loading:** reads images, or the first page of a PDF at 300 DPI via `pdf2image`, and converts them to grayscale.
2. **Deskewing:** detects the skew angle with `minAreaRect` and straightens the scan if the angle is between 0.3° and 15°.
3. **Title-block crop:** keeps the bottom 45 % and right 60 %, applies a median blur and upscales to about 3500 px (at most 4×).
4. **Four binarisation variants:** CLAHE with adaptive threshold, bilateral filter with Otsu, table-line removal, and a morphologically thickened version.
5. **Tesseract** runs on each variant with page segmentation modes 11, 6 and 4. That is up to 12 passes, using the languages `deu+eng+Fraktur`. Words with a confidence below 20 are dropped.
6. **Field parsing:**
   - *Scale:* the regex `1:x`, with OCR digit fixes (`O→0`, `I→1`, `S→5` …). The value is only accepted if it is a common scale.
   - *Date:* `DD.MM.YYYY` or ISO format, with years between 1890 and the next year. The **latest** valid date is taken.
   - *ID:* regex patterns for typical BVG and DIN plan numbers.
   - *Title:* fallback heuristic. It picks the line with the largest `text height × confidence` that is not a label and contains no date.
7. For each field, the result with the highest priority across all passes is kept.
- Output: `results_ocr_v4.txt`, `results_ocr_v4.csv` (including the number of passes and the runtime per file) and the preprocessed images in `debug_v4/`.

`old_data/` contains the earlier development stages v1–v3. They cover a simple crop with `image_to_string`, OpenCV preprocessing, and spatial label-to-value assignment with a multi-stage cascade.

---

## Evaluation: `evaluate_results.py`

```bash
python evaluate_results.py --results results_api.txt --gt groundtruth.xlsx [--model qwen3.8-27b] [--out-dir eval_output]
```

**Ground truth format** (`.xlsx` first sheet, or `.csv`): columns `File Name | Title | Date | ID | Scale`. Multiple valid answers go in one cell, separated by `|` (e.g. `1:100|1:50`). An empty cell or `Not found` means the field does not exist on the plan. File names are matched without their extension. For multi-page PDFs, a value counts as correct if **any** page's prediction matches.

### Step 1: Strict comparison (no LLM)
Both values are normalised first and then compared:

| Field | Normalisation | Match if |
|-------|---------------|----------|
| Title | Unicode NFKC, lowercase, no punctuation | Similarity ≥ 0.85 (`SequenceMatcher`: max of plain and word-sorted comparison) |
| ID | Lowercase, remove spaces, `- _ / . , : ;` | Identical (1.0) |
| Date | Convert to ISO (`YYYY-MM-DD` / `YYYY-MM` / `YYYY`); supports `DD.MM.YY(YY)`, German and English month names, 2-digit years | Identical |
| Scale | Extract all ratios `a:b` or `a/b` | At least one ratio in common |

### Step 2: Fair evaluation (LLM judge)
If a strict comparison fails:
- **Date and Scale** are checked deterministically. A date that matches but is less precise (e.g. only the year instead of the full date) is rated **PARTIAL**.
- **Title and ID** go to an **LLM judge**: the Qwen model on the university API (`qwen3.8-27b` by default, thinking mode). It answers **YES / PARTIAL / NO** with a short reason, following field-specific rules. Spelling, abbreviations and word order are tolerated, but a different object or different digits count as NO.
- Safeguard: if the judge rates an ID as YES or PARTIAL but the character similarity is below 0.5, the verdict is overridden to NO.
- The judge's answers are cached, so an identical comparison is only sent once.

### Classification of each field instance
| Category | Meaning |
|----------|---------|
| `TN` | Ground truth and prediction are both "Not found" (correct) |
| `FP` | The model returns a value where there is none |
| `FN` | The model finds nothing although a value exists |
| `MATCH_STRICT` | Strict match |
| `MATCH_LLM` | Judge verdict YES |
| `PARTIAL` | Judge verdict PARTIAL |
| `WRONG` | Judge verdict NO |

### Computed metrics (per field, micro over all fields, macro average of the fields)
| Metric | Formula |
|--------|---------|
| **Strict accuracy** | (TN + MATCH_STRICT) / N |
| **Strict precision / recall / F1** | TP = MATCH_STRICT. Every other wrong value counts as both FP and FN |
| **Fair accuracy** | (TN + MATCH_STRICT + MATCH_LLM) / N |
| **Soft score** | (TN + MATCH_STRICT + MATCH_LLM + 0.5·PARTIAL) / N. This is the headline figure, reported as *overall agreement* |
| **Fair precision / recall / F1** | TP = MATCH_STRICT + MATCH_LLM. PARTIAL and WRONG count as FP and FN |
| **Fully correct plans** | Number of plans with all 4 fields correct (strict and fair) |

Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean.

**Output** in `eval_output/`:
- `metrics_<timestamp>.txt` / `.csv`: summary tables
- `details_<timestamp>.csv`: every single decision with ground truth, prediction, category and the judge's reason

---

## Setup

```bash
pip install -r requirements.txt
```

- **Tesseract** must be installed on the system (`brew install tesseract tesseract-lang`), including the `deu` and Fraktur models. `pdf2image` also needs **Poppler** (`brew install poppler`).
- **Ollama** is needed for the local model: `ollama pull qwen2.5vl:3b`.
- **`.env`** goes in the project folder and must not be committed:
  ```
  HTW_API_KEY=...
  HTW_BASE_URL=...
  MODEL_NAME=...
  ```

Plan images and `.env` are excluded via `.gitignore`.
