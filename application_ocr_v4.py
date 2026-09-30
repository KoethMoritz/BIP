"""
OCR v4 - Heavy-Duty Extrahierung für historische BVG-Baupläne & Zeichnungen
(Inklusive nativer PDF-Unterstützung)

Highlights:
  * Unterstützt Bilder (.tif, .png, .jpg etc.) UND PDFs (.pdf)
  * Automatisches Deskewing (Begradigung schräger Scans)
  * Ensemble aus 4 OpenCV-Binarisierungsverfahren
  * Tesseract-Word-Confidence Scoring & Fuzzy-Label-Matching
  * Export in TXT, CSV und Debug-Archiv
"""

import csv
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Für PDF-Verarbeitung
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

Image.MAX_IMAGE_PIXELS = None  # Sehr große Scans (A0/A1) erlauben

# ============================================================
# KONFIGURATION
# ============================================================

OUTPUT_TXT = "results_ocr_v4.txt"
OUTPUT_CSV = "results_ocr_v4.csv"
DEBUG_DIR = "debug_v4"
SAVE_DEBUG = True

# Ergänzt um .pdf
SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf")
OCR_LANG = "deu+eng+script/Fraktur+deu_latf"
MIN_WORD_CONF = 20
TARGET_LONG_SIDE = 3500
MAX_UPSCALE = 4.0

FIELDS = ("Title", "Date", "ID", "Scale")

KNOWN_SCALES = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000}

# Typische BVG / DIN Plannummern-Muster
ID_PATTERNS = [
    re.compile(r"\b[A-Z]{1,5}[_\-\s/]{1,2}\d{2,6}(?:[_\-\s/]{1,2}[A-Z0-9]{1,8}){1,5}\b", re.I),
    re.compile(r"\b\d{3,6}(?:[_\-\s/]{1,2}\d{1,6}){1,5}[A-Z]?\b", re.I),
    re.compile(r"\bBVG[_\-\s]?[A-Z0-9]{2,12}\b", re.I),
]

# Fuzzy-fähige Labels
LABELS = {
    "Title": [
        r"planinhalt|plantitel|zeichnungstitel|\btitel\b|\btitle\b|bezeichnung|bauvorhaben|\bprojekt\b|\bobjekt\b|strecke|station|bauwerk",
    ],
    "Date": [r"\bdatum\b|\bdate\b|ausgabe(?:datum)?|\bstand\b|gez(?:eichnet)?|gepr(?:üft)?"],
    "ID": [
        r"plan\s?-?\s?(?:nr|nummer|no)\b\.?|zeichnungs\s?-?\s?(?:nr|nummer)\b\.?|dok(?:ument)?\.?\s?-?\s?(?:nr|nummer)\b\.?|blatt\s?-?\s?(?:nr|nummer)\b\.?"
    ],
    "Scale": [r"\bma\S{0,3}stab\b|\bmst\b\.?|\bscale\b"],
}

EXTRA_STOP = (
    r"\bindex\b|\bformat\b|bearbeiter|auftraggeber|änderung|aenderung|\bgebäude\b|\banlage\b|\bblatt\b"
)

LABEL_RES = {k: [re.compile(p, re.I) for p in v] for k, v in LABELS.items()}
STOP_RE = re.compile("|".join(p for v in LABELS.values() for p in v) + "|" + EXTRA_STOP, re.I)

DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4}|\d{2})(?!\d)")
ISO_RE = re.compile(r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)")
SCALE_RE = re.compile(r"(?<!\d)1\s*[:;]\s*([0-9OoIlSB]{1,5})(?!\d)")
_OCR_DIGITS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


# ============================================================
# TESSERACT SETUP
# ============================================================

def setup_tesseract():
    if platform.system() == "Darwin":
        cmd = shutil.which("tesseract")
        for p in (cmd, "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
            if p and os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    elif platform.system() == "Windows":
        for p in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break


# ============================================================
# BILD- & PDF-LADE-LOGIK
# ============================================================

def load_gray_from_file(path: str) -> np.ndarray:
    """Lädt Bilder sowie PDFs und wandelt sie in ein Graustufen-Numpy-Array um."""
    if path.lower().endswith(".pdf"):
        if not PDF_SUPPORT:
            raise ImportError("Paket 'pdf2image' ist nicht installiert. Bitte 'pip install pdf2image' ausführen.")

        # Rendere die erste Seite der PDF mit 300 DPI für hohe OCR-Genauigkeit
        pages = convert_from_path(path, first_page=1, last_page=1, dpi=300)
        if not pages:
            raise ValueError("PDF konnte nicht gelesen werden oder ist leer.")
        pil_img = pages[0].convert("L")
        return np.array(pil_img)
    else:
        with Image.open(path) as img:
            img.seek(0)
            if img.mode.startswith("I"):
                arr = np.array(img).astype(np.float32)
                lo, hi = float(arr.min()), float(arr.max())
                arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
                return arr.astype(np.uint8)
            return np.array(img.convert("L"))


def deskew(gray: np.ndarray) -> np.ndarray:
    """Richtet leicht schief eingescannte Pläne automatisch aus."""
    try:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if 0.3 < abs(angle) < 15.0:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return gray


def remove_table_lines(gray: np.ndarray) -> np.ndarray:
    """Entfernt störende Gitter- und Tabellenlinien des Plankopfs."""
    binv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    h, w = gray.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 30), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 30)))
    lines = cv2.add(
        cv2.morphologyEx(binv, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(binv, cv2.MORPH_OPEN, vk),
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    cleaned = gray.copy()
    cleaned[lines > 0] = 255
    return cleaned


def preprocess_variants(crop: np.ndarray) -> dict:
    """Erzeugt 4 spezialisierte Binarisierungsvarianten für historische Scans."""
    variants = {}

    # 1. CLAHE + Adaptive Thresholding
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(crop)
    variants["adaptive"] = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )

    # 2. Bilateral Filter + Otsu
    blur = cv2.bilateralFilter(crop, 9, 75, 75)
    _, variants["otsu"] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Cleaned No-Lines
    nolines_gray = remove_table_lines(clahe)
    variants["nolines"] = cv2.adaptiveThreshold(
        nolines_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 12
    )

    # 4. Dilated / Morph-Enhanced
    kernel = np.ones((2, 2), np.uint8)
    variants["morph_thick"] = cv2.erode(variants["otsu"], kernel, iterations=1)

    return variants


# ============================================================
# METADATEN-EXTRAKTIONS-LOGIK
# ============================================================

@dataclass
class Line:
    text: str
    conf: float
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def h(self) -> int:
        return max(self.y1 - self.y0, 1)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def ocr_lines_with_conf(img: np.ndarray, psm: int) -> list:
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    d = pytesseract.image_to_data(
        img, lang=OCR_LANG, config=config, output_type=pytesseract.Output.DICT
    )
    groups = {}
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        if not txt:
            continue
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf < MIN_WORD_CONF:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        groups.setdefault(key, []).append(
            (d["left"][i], d["top"][i], d["width"][i], d["height"][i], txt, conf)
        )
    lines = []
    for words in groups.values():
        words.sort(key=lambda w: w[0])
        avg_conf = sum(w[5] for w in words) / len(words)
        lines.append(Line(
            text=" ".join(w[4] for w in words),
            conf=avg_conf,
            x0=min(w[0] for w in words),
            y0=min(w[1] for w in words),
            x1=max(w[0] + w[2] for w in words),
            y1=max(w[1] + w[3] for w in words),
        ))
    lines.sort(key=lambda l: (l.y0, l.x0))
    return lines


def parse_scale(text: str):
    for m in SCALE_RE.finditer(text):
        digits = m.group(1).translate(_OCR_DIGITS)
        if digits.isdigit() and int(digits) in KNOWN_SCALES:
            return f"1:{int(digits)}"
    return None


def parse_dates(text: str) -> list:
    now = datetime.now().year
    found = []

    def add(y, mo, d, m):
        if not (1890 <= y <= now + 1):
            return
        try:
            dt = datetime(y, mo, d)
            found.append((dt, re.sub(r"\s+", "", m.group(0))))
        except ValueError:
            pass

    for m in DATE_RE.finditer(text):
        y = int(m.group(3))
        if len(m.group(3)) == 2:
            y += 2000 if y <= (now % 100) + 1 else 1900
        add(y, int(m.group(2)), int(m.group(1)), m)
    for m in ISO_RE.finditer(text):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)), m)
    return found


def extract_fields_v4(lines) -> dict:
    text = "\n".join(l.text for l in lines)
    res = {}

    # Scale
    sc = parse_scale(text)
    if sc:
        res["Scale"] = (2, sc)

    # Date
    dates = parse_dates(text)
    if dates:
        res["Date"] = (2, max(dates, key=lambda d: d[0])[1])

    # ID
    clean_text = DATE_RE.sub(" ", text)
    for pat in ID_PATTERNS:
        m = pat.search(clean_text)
        if m:
            res["ID"] = (2, m.group(0))
            break

    # Title Fallback über Schrifthöhe / Wörter
    cands = [
        l for l in lines
        if len(l.text) >= 6 and not STOP_RE.search(l.text) and not DATE_RE.search(l.text)
    ]
    if cands:
        best_cand = max(cands, key=lambda l: (l.h * l.conf, len(l.text)))
        res["Title"] = (1, best_cand.text)

    return res


# ============================================================
# PROCESSING PIPELINE
# ============================================================

def process_file_v4(path: str, stem: str, debug_dir: Path):
    raw_gray = load_gray_from_file(path)
    gray = deskew(raw_gray)

    h, w = gray.shape
    # Schriftfeld Ausschnitt (rechts unten)
    crop_gray = gray[int(h * 0.55):h, int(w * 0.40):w]
    crop_gray = cv2.medianBlur(crop_gray, 3)

    scale = min(TARGET_LONG_SIDE / max(crop_gray.shape), MAX_UPSCALE)
    if scale > 1.0:
        crop_gray = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    variants = preprocess_variants(crop_gray)
    best_results = {}
    passes_used = 0

    for var_name, var_img in variants.items():
        for psm in (11, 6, 4):
            passes_used += 1
            lines = ocr_lines_with_conf(var_img, psm)

            if SAVE_DEBUG:
                tag = f"{stem}_var_{var_name}_psm{psm}"
                cv2.imwrite(str(debug_dir / f"{tag}.png"), var_img)

            fields = extract_fields_v4(lines)
            for k, v in fields.items():
                if k not in best_results or v[0] > best_results[k][0]:
                    best_results[k] = v

            if all(k in best_results and best_results[k][0] == 2 for k in FIELDS):
                break

    final_dict = {k: (best_results[k][1] if k in best_results else "Not found") for k in FIELDS}
    return final_dict, passes_used


def main():
    setup_tesseract()
    script_dir = Path(__file__).resolve().parent
    txt_path, csv_path = script_dir / OUTPUT_TXT, script_dir / OUTPUT_CSV
    debug_dir = script_dir / DEBUG_DIR
    if SAVE_DEBUG:
        debug_dir.mkdir(exist_ok=True)

    folder = input("Enter the path to the folder with images/PDFs: ").strip().strip('"\'')
    if not os.path.isdir(folder):
        sys.exit(f"Error: Folder does not exist -> {folder}")

    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not files:
        sys.exit(f"No compatible images/PDFs found in: {folder}")

    print(f"\nFound {len(files)} files. Starting Heavy-Duty OCR v4...")
    print(f"Results: {txt_path} & {csv_path}\n")

    with open(txt_path, "w", encoding="utf-8") as out, \
            open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "title", "date", "id", "scale", "passes_used", "seconds"])

        for idx, filename in enumerate(files, start=1):
            print(f"[{idx}/{len(files)}] Processing: {filename}...")
            t0 = time.perf_counter()
            try:
                r, passes = process_file_v4(os.path.join(folder, filename), Path(filename).stem, debug_dir)
                result_text = "\n".join(f"{k}: {r[k]}" for k in FIELDS)
                writer.writerow([filename, r["Title"], r["Date"], r["ID"], r["Scale"],
                                 passes, f"{time.perf_counter() - t0:.1f}"])
            except Exception as e:
                result_text = f"Error processing file: {e}"
                writer.writerow([filename, "", "", "", "", "ERROR", str(e)])
                print(f"   -> Failed: {e}")

            out.write(f"FILE: {filename}\n{'-' * 40}\n{result_text}\n{'=' * 60}\n\n")
            out.flush()
            fcsv.flush()
            print(f"   -> Done ({time.perf_counter() - t0:.1f}s).")

    print(f"\nFinished! Check '{txt_path}' and '{csv_path}'.")


if __name__ == "__main__":
    main()