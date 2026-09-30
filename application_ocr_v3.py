"""
OCR v3 - Schriftfeld-Extraktion fuer Bauplaene (BVG) mit PDF-Unterstuetzung

Aenderungen gegenueber v2:
  * Unterstuetzung fuer PDF-Dateien (via pypdfium2, seitenweise, 300 DPI)
  * Crop wird in NATIVER Aufloesung ausgeschnitten, erst danach skaliert
  * Mehrstufige Pipeline (Cascade): mehrere Crops / Vorverarbeitungen / PSM-Modi,
    fehlende Felder werden aus spaeteren Durchlaeufen aufgefuellt
  * Tabellenlinien des Schriftfelds werden entfernt (stoeren Tesseract stark)
  * Keine harte Otsu-Binarisierung mehr (CLAHE auf Graustufen)
  * image_to_data statt image_to_string -> Bounding Boxes
  * Raeumliche Label->Wert-Zuordnung (rechts vom Label / darunter)
  * Validierung + OCR-Fehlerkorrektur (Scale gegen Liste ueblicher Massstaebe,
    Datum wird wirklich geparst)
  * Label-Treffer haben Vorrang vor heuristischen Fallbacks
  * Debug-Ausgabe (Crop-Bild + OCR-Zeilen mit Koordinaten) und CSV zusaetzlich zur TXT
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
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # sehr grosse Scans erlauben

# ============================================================
# CONFIG
# ============================================================

OUTPUT_TXT = "results_ocr_v3.txt"
OUTPUT_CSV = "results_ocr_v3.csv"
DEBUG_DIR = "debug_v3"
SAVE_DEBUG = True  # speichert Crops + OCR-Zeilen pro Datei -> zum Fehler-Analysieren
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
)

OCR_LANG = "deu+eng"
MIN_WORD_CONF = 30        # Woerter mit geringerer Tesseract-Konfidenz werden verworfen
TARGET_LONG_SIDE = 5000   # Crop wird auf diese lange Seite skaliert (hoch- oder runter)
MAX_UPSCALE = 3.0

# Crop-Kandidaten als Anteil der Bildgroesse: (x0, y0, x1, y1)
CROPS = {
    "br_small": (0.55, 0.70, 1.00, 1.00),
    "br_large": (0.35, 0.50, 1.00, 1.00),
}

# Durchlaeufe in Reihenfolge: (Crop, Vorverarbeitung, Tesseract-PSM)
# Es wird gestoppt, sobald alle 4 Felder ueber ein Label gefunden wurden.
PASSES = [
    ("br_small", "gray", 11),
    ("br_small", "nolines", 11),
    ("br_large", "nolines", 11),
    ("br_large", "gray", 12),
]

FIELDS = ("Title", "Date", "ID", "Scale")

# Ueblicher Massstaebe (Nenner). Alles andere wird als OCR-Fehler verworfen.
KNOWN_SCALES = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000}

# BVG-Plannummern Heuristiken (Beispiele: S_510_023 / 1234-05-001)
ID_PATTERNS = [
    re.compile(r"\b[A-Z]{1,4}[_\-]\d{2,5}(?:[_\-][A-Z0-9]{1,6}){1,4}\b"),
    re.compile(r"\b\d{3,}(?:[_\-/]\d{1,6}){1,4}[A-Z]?\b"),
]

# Label-Regexe je Feld, in Prioritaetsreihenfolge
LABELS = {
    "Title": [
        r"planinhalt|plantitel|zeichnungstitel|\btitel\b|\btitle\b|bezeichnung",
        r"bauvorhaben|\bprojekt\b|\bobjekt\b",
    ],
    "Date": [r"\bdatum\b|\bdate\b|ausgabe(?:datum)?|\bstand\b"],
    "ID": [
        r"plan\s?-?\s?(?:nr|nummer|no)\b\.?|zeichnungs\s?-?\s?(?:nr|nummer)\b\.?"
        r"|dok(?:ument)?\.?\s?-?\s?(?:nr|nummer)\b\.?|drawing\s?(?:no|number)\b\.?"
    ],
    "Scale": [r"\bma\S{0,2}stab\b|\bmst\b\.?|\bscale\b"],
}

EXTRA_STOP = (
    r"\bgez(?:eichnet)?\b|\bgepr(?:uef|ü)?(?:t|ft)?\b|\bindex\b|\bblatt\b|\bformat\b"
    r"|bearbeiter|auftraggeber|änderung|aenderung|\bgebäude\b|\bgebaeude\b|\banlage\b"
)

LABEL_RES = {k: [re.compile(p, re.I) for p in v] for k, v in LABELS.items()}
STOP_RE = re.compile(
    "|".join(p for v in LABELS.values() for p in v) + "|" + EXTRA_STOP, re.I
)

DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4}|\d{2})(?!\d)"
)
ISO_RE = re.compile(
    r"(?<!\d)(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)"
)
SCALE_RE = re.compile(r"(?<!\d)1\s*[:;]\s*([0-9OoIlSB]{1,5})(?!\d)")
_OCR_DIGITS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


# ============================================================
# TESSERACT PFAD (macOS / Windows)
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

    global OCR_LANG
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as e:
        sys.exit(f"Tesseract nicht gefunden/lauffaehig: {e}\n"
                 f"macOS: brew install tesseract tesseract-lang")
    if "deu" not in available and "deu" in OCR_LANG:
        print("WARNUNG: Sprachpaket 'deu' fehlt (brew install tesseract-lang). Nutze nur 'eng'.")
        OCR_LANG = "eng"


# ============================================================
# BILD / PDF LADEN & VORVERARBEITEN
# ============================================================

def pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """Konvertiert ein PIL-Image in ein uint8-Graustufenarray."""
    if img.mode.startswith("I"):  # 16/32-bit Graustufen
        arr = np.array(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
        return arr.astype(np.uint8)
    return np.array(img.convert("L"))


def load_gray_pages(path: str) -> list[tuple[str, np.ndarray]]:
    """Laedt Bild- oder PDF-Seiten als uint8-Graustufenarrays.
    Gibt Liste aus: [(page_suffix, gray_array), ...] zurueck.
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        pages = []
        pdf = pdfium.PdfDocument(path)
        total_pages = len(pdf)

        for page_index in range(total_pages):
            page = pdf[page_index]
            # 300 DPI Rendering fuer exakte OCR-Kanten
            bitmap = page.render(scale=300 / 72)
            pil_img = bitmap.to_pil()
            gray = pil_to_gray_array(pil_img)
            suffix = f"_page{page_index + 1}" if total_pages > 1 else ""
            pages.append((suffix, gray))

        return pages
    else:
        with Image.open(path) as img:
            img.seek(0)
            return [("", pil_to_gray_array(img))]


def remove_table_lines(gray: np.ndarray) -> np.ndarray:
    """Entfernt lange horizontale/vertikale Linien (Tabellenraster des Schriftfelds)."""
    binv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )
    h, w = gray.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 25)))
    lines = cv2.add(
        cv2.morphologyEx(binv, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(binv, cv2.MORPH_OPEN, vk),
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    cleaned = gray.copy()
    cleaned[lines > 0] = 255
    return cleaned


def prepare(gray: np.ndarray, crop_name: str, mode: str) -> np.ndarray:
    h, w = gray.shape
    x0, y0, x1, y1 = CROPS[crop_name]
    # 1) In NATIVER Aufloesung croppen
    crop = np.ascontiguousarray(gray[int(h * y0):int(h * y1), int(w * x0):int(w * x1)])
    # 2) Erst danach auf Zielgroesse skalieren
    scale = min(TARGET_LONG_SIDE / max(crop.shape), MAX_UPSCALE)
    if abs(scale - 1) > 0.05:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
    # 3) Optional Linien entfernen
    if mode == "nolines":
        crop = remove_table_lines(crop)
    # 4) Kontrastverstaerkung per CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(crop)


# ============================================================
# OCR -> ZEILEN MIT KOORDINATEN
# ============================================================

@dataclass
class Line:
    text: str
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


def ocr_lines(img: np.ndarray, psm: int) -> list:
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
            (d["left"][i], d["top"][i], d["width"][i], d["height"][i], txt)
        )
    lines = []
    for words in groups.values():
        words.sort(key=lambda w: w[0])
        lines.append(Line(
            text=" ".join(w[4] for w in words),
            x0=min(w[0] for w in words),
            y0=min(w[1] for w in words),
            x1=max(w[0] + w[2] for w in words),
            y1=max(w[1] + w[3] for w in words),
        ))
    lines.sort(key=lambda l: (l.y0, l.x0))
    return lines


# ============================================================
# RAEUMLICHE HILFSFUNKTIONEN
# ============================================================

def clean_value(s: str) -> str:
    return s.strip(" :.;-_|\t")


def alpha_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def right_of(lines, i):
    """Zeilen in derselben Reihe rechts vom Label (Wert in Nachbarzelle)."""
    lab = lines[i]
    out = []
    for j, l in enumerate(lines):
        if j == i or STOP_RE.search(l.text):
            continue
        same_row = abs(l.cy - lab.cy) <= 0.6 * max(lab.h, l.h)
        gap = l.x0 - lab.x1
        if same_row and -5 <= gap <= 15 * lab.h:
            out.append(l)
    return sorted(out, key=lambda l: l.x0)


def below_of(lines, i):
    """Zeilen direkt unter dem Label (typisch: kleines Label oben, Wert darunter)."""
    lab = lines[i]
    cand = [
        l for j, l in enumerate(lines)
        if j != i and l.y0 >= lab.y1 - 0.3 * lab.h
        and l.x0 <= lab.x1 and l.x1 >= lab.x0
    ]
    cand.sort(key=lambda l: l.y0)
    out, prev_bottom = [], lab.y1
    for l in cand:
        if l.y0 - prev_bottom > 2.0 * max(lab.h, l.h):
            break
        if STOP_RE.search(l.text):
            break
        out.append(l)
        prev_bottom = l.y1
    return out


def candidate_texts(lines, i, m_end):
    lab = lines[i]
    out = []
    rest = clean_value(lab.text[m_end:])
    if rest:
        out.append(rest)
    out += [l.text for l in right_of(lines, i)]
    out += [l.text for l in below_of(lines, i)]
    return out


def find_labeled(lines, key, parse):
    for pat in LABEL_RES[key]:
        for i, ln in enumerate(lines):
            m = pat.search(ln.text)
            if not m:
                continue
            for cand in candidate_texts(lines, i, m.end()):
                val = parse(cand)
                if val:
                    return val
    return None


# ============================================================
# PARSER / VALIDATOREN
# ============================================================

def parse_scale(text: str):
    for m in SCALE_RE.finditer(text):
        digits = m.group(1).translate(_OCR_DIGITS)
        if digits.isdigit() and int(digits) in KNOWN_SCALES:
            return f"1:{int(digits)}"
    return None


def parse_dates(text: str) -> list:
    """Gibt [(datetime, roher_string)] fuer alle gueltigen Datumsangaben zurueck."""
    now = datetime.now().year
    found = []

    def add(y, mo, d, m):
        if not (1950 <= y <= now + 1):
            return
        try:
            dt = datetime(y, mo, d)
        except ValueError:
            return
        found.append((dt, re.sub(r"\s+", "", m.group(0))))

    for m in DATE_RE.finditer(text):
        y = int(m.group(3))
        if len(m.group(3)) == 2:
            y += 2000 if y <= (now % 100) + 1 else 1900
        add(y, int(m.group(2)), int(m.group(1)), m)
    for m in ISO_RE.finditer(text):
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)), m)
    return found


def parse_date_first(text: str):
    d = parse_dates(text)
    return d[0][1] if d else None


def clean_id(s: str) -> str:
    s = s.strip(" :.-_|")
    return re.sub(r"\s*([_\-/])\s*", r"\1", s)


def parse_id(text: str):
    text = DATE_RE.sub(" ", text)
    for pat in ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    s = clean_id(text)
    if (3 <= len(s) <= 40 and re.search(r"\d", s) and len(s.split()) <= 2
            and not STOP_RE.search(s)):
        return s
    return None


def title_from_label(lines):
    for pat in LABEL_RES["Title"]:
        for i, ln in enumerate(lines):
            m = pat.search(ln.text)
            if not m:
                continue
            parts = []
            rest = clean_value(ln.text[m.end():])
            if len(rest) >= 3:
                parts.append(rest)
            else:
                r = right_of(lines, i)
                if r:
                    parts.append(r[0].text)
            parts += [l.text for l in below_of(lines, i)[:2]]
            parts = [p for p in parts if alpha_ratio(p) >= 0.5]
            title = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if len(title) >= 4:
                return title
    return None


def guess_title(lines):
    """Fallback: Titel ist meist die Zeile mit der groessten Schrift."""
    cands = [
        l for l in lines
        if len(l.text) >= 8 and alpha_ratio(l.text) >= 0.6
        and not STOP_RE.search(l.text) and not DATE_RE.search(l.text)
    ]
    return max(cands, key=lambda l: (l.h, len(l.text))).text if cands else None


# ============================================================
# FELDER EXTRAHIEREN (Strength 2 = Label-Treffer, 1 = Fallback)
# ============================================================

def extract_fields(lines) -> dict:
    text = "\n".join(l.text for l in lines)
    res = {}

    v = find_labeled(lines, "Scale", parse_scale)
    if v:
        res["Scale"] = (2, v)
    elif parse_scale(text):
        res["Scale"] = (1, parse_scale(text))

    v = find_labeled(lines, "Date", parse_date_first)
    if v:
        res["Date"] = (2, v)
    else:
        dates = parse_dates(text)
        if dates:
            res["Date"] = (1, max(dates, key=lambda d: d[0])[1])

    v = find_labeled(lines, "ID", parse_id)
    if v:
        res["ID"] = (2, v)
    else:
        clean_text = DATE_RE.sub(" ", text)
        for pat in ID_PATTERNS:
            m = pat.search(clean_text)
            if m:
                res["ID"] = (1, m.group(0))
                break

    v = title_from_label(lines)
    if v:
        res["Title"] = (2, v)
    else:
        v = guess_title(lines)
        if v:
            res["Title"] = (1, v)
    return res


# ============================================================
# VERARBEITUNG EINES GRAUSTUFEN-ARRAYS (EINZELNE SEITE)
# ============================================================

def process_single_gray_image(gray: np.ndarray, stem: str, debug_dir: Path):
    best, passes_run = {}, 0
    for n, (crop_name, mode, psm) in enumerate(PASSES, start=1):
        img = prepare(gray, crop_name, mode)
        lines = ocr_lines(img, psm)
        passes_run = n

        if SAVE_DEBUG:
            tag = f"{stem}_p{n}_{crop_name}_{mode}_psm{psm}"
            cv2.imwrite(str(debug_dir / f"{tag}.png"), img)
            (debug_dir / f"{tag}.txt").write_text(
                "\n".join(f"{l.x0},{l.y0}\t{l.text}" for l in lines), encoding="utf-8"
            )

        for k, v in extract_fields(lines).items():
            if k not in best or v[0] > best[k][0]:
                best[k] = v
        if all(k in best and best[k][0] == 2 for k in FIELDS):
            break

    result = {k: (best[k][1] if k in best else "Not found") for k in FIELDS}
    return result, passes_run


# ============================================================
# MAIN
# ============================================================

def main():
    setup_tesseract()
    script_dir = Path(__file__).resolve().parent
    txt_path, csv_path = script_dir / OUTPUT_TXT, script_dir / OUTPUT_CSV
    debug_dir = script_dir / DEBUG_DIR
    if SAVE_DEBUG:
        debug_dir.mkdir(exist_ok=True)

    folder = input("Enter the path to the folder with files (images/PDFs): ").strip().strip('"\'')
    if not os.path.isdir(folder):
        sys.exit(f"Error: Folder does not exist -> {folder}")

    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not files:
        sys.exit(f"No compatible files found in: {folder}")

    print(f"\nFound {len(files)} files. Starting OCR v3...")
    print(f"Results: {txt_path}\n")

    with open(txt_path, "w", encoding="utf-8") as out, \
         open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "title", "date", "id", "scale", "passes_used", "seconds"])

        for idx, filename in enumerate(files, start=1):
            file_path = os.path.join(folder, filename)
            base_stem = Path(filename).stem
            print(f"[{idx}/{len(files)}] Processing: {filename}...")

            try:
                pages = load_gray_pages(file_path)

                for page_suffix, gray_array in pages:
                    t0 = time.perf_counter()
                    item_display = f"{filename}{page_suffix}"
                    tag_stem = f"{base_stem}{page_suffix}"

                    if page_suffix:
                        print(f"   -> Analyzing page {page_suffix.replace('_page', '')}...")

                    r, passes = process_single_gray_image(gray_array, tag_stem, debug_dir)
                    elapsed = time.perf_counter() - t0

                    result_text = "\n".join(f"{k}: {r[k]}" for k in FIELDS)
                    writer.writerow([
                        item_display, r["Title"], r["Date"], r["ID"], r["Scale"],
                        passes, f"{elapsed:.1f}"
                    ])

                    out.write(f"FILE: {item_display}\n{'-' * 40}\n{result_text}\n{'=' * 60}\n\n")
                    out.flush()
                    fcsv.flush()
                    print(f"   -> Done {item_display} ({elapsed:.1f}s).")

            except Exception as e:
                result_text = f"Error processing file: {e}"
                writer.writerow([filename, "", "", "", "", "ERROR", str(e)])
                out.write(f"FILE: {filename}\n{'-' * 40}\n{result_text}\n{'=' * 60}\n\n")
                out.flush()
                fcsv.flush()
                print(f"   -> Failed: {e}")

    print(f"\nFinished! See '{txt_path}' and '{csv_path}'.")


if __name__ == "__main__":
    main()