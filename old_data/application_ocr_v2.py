import os
import platform
import re
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
import pytesseract

# Prevent DecompressionBombWarning for very large scanned files
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

OUTPUT_FILE = "results_ocr_v2.txt"
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
)

# Fokusbereich: Untere rechte Ecke (Schriftfeld nach DIN EN ISO 7200)
FOCUS_BOTTOM_RIGHT = True

# Tesseract Page Segmentation Mode:
# 11 = Find as much text as possible in no particular order (ideal für verteilte Schriftfelder)
# 6  = Uniform block of text
TESSERACT_CONFIG = r"--oem 3 --psm 11"

# ============================================================
# TESSERACT BINARY PATH DETECTION
# ============================================================

if platform.system() == "Darwin":
    system_tesseract = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    if os.path.exists(system_tesseract):
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
    elif os.path.exists("/usr/local/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

elif platform.system() == "Windows":
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in win_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

script_dir = Path(__file__).resolve().parent
output_file_path = script_dir / OUTPUT_FILE


# ============================================================
# IMAGE PREPROCESSING (OpenCV Pipeline)
# ============================================================

def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    """Bereitet den Scan für technische OCR auf:
    Graustufen -> Kontrastverstärkung (CLAHE) -> Binarisierung.
    """
    img = np.array(pil_img)

    # 1. In Graustufen umwandeln
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Leichtes Denoising & Otsu Thresholding für scharfe Kanten
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


# ============================================================
# METADATA EXTRACTION LOGIC (v2 - refined regex)
# ============================================================

def extract_metadata_v2(raw_text: str) -> dict:
    metadata = {
        "Title": "Not found",
        "Date": "Not found",
        "ID": "Not found",
        "Scale": "Not found",
    }

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Scale
    scale_pattern = re.compile(
        r"(?:(?:M|Mst|Maßstab|Scale)\s*[:\.]?\s*)?(1\s*:\s*\d{1,4}(?:/\d{1,4})?|1/\d+[\"']?\s*=\s*\d+)",
        re.IGNORECASE,
    )

    # 2. Date
    date_pattern = re.compile(
        r"(?:(?:Datum|Date|gez|gepr)\.?\s*[:\.]?\s*)?(\b\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}\b|\b\d{4}[\.\/\-]\d{2}[\.\/\-]\d{2}\b)",
        re.IGNORECASE,
    )

    # 3. ID / Plannummer
    id_label_pattern = re.compile(
        r"(?:plan(?:-|\s*)?nr\.?|zeichnungs(?:-|\s*)?nr\.?|dok(?:ument)?(?:-|\s*)?nr\.?|drawing\s*no\.?|id|code|blatt-?nr\.?)\s*[:\.]?\s*([A-Z0-9\-_./]+)",
        re.IGNORECASE,
    )
    standalone_id_pattern = re.compile(r"\b([A-Z]{1,3}[_\-]\d{2,4}[_\-][A-Z0-9]+)\b")

    # 4. Title Keywords
    title_label_pattern = re.compile(
        r"(?:titel|bezeichnung|bauvorhaben|planinhalt|projekt|object|title)\s*[:\.]?\s*(.*)",
        re.IGNORECASE,
    )

    candidate_titles = []

    for i, line in enumerate(lines):
        # Scale
        if metadata["Scale"] == "Not found":
            m_scale = scale_pattern.search(line)
            if m_scale:
                metadata["Scale"] = m_scale.group(1).replace(" ", "")

        # Date
        if metadata["Date"] == "Not found":
            m_date = date_pattern.search(line)
            if m_date:
                metadata["Date"] = m_date.group(1)

        # ID
        if metadata["ID"] == "Not found":
            m_id = id_label_pattern.search(line)
            if m_id and len(m_id.group(1).strip()) > 2:
                metadata["ID"] = m_id.group(1).strip()
            elif any(k in line.lower() for k in ["plan-nr", "plannr", "zeichnungsnr"]) and i + 1 < len(lines):
                next_val = lines[i + 1].strip()
                if 3 < len(next_val) < 40 and not any(s in next_val.lower() for s in ["maßstab", "datum", "gezeichnet"]):
                    metadata["ID"] = next_val
            else:
                m_stand = standalone_id_pattern.search(line)
                if m_stand:
                    metadata["ID"] = m_stand.group(1)

        # Title
        if metadata["Title"] == "Not found":
            m_title = title_label_pattern.search(line)
            if m_title and len(m_title.group(1).strip()) > 3:
                metadata["Title"] = m_title.group(1).strip()

        # Zeilen für Title-Fallback
        ignore_words = ["datum", "maßstab", "scale", "gez", "gepr", "index", "format"]
        if len(line) > 5 and not any(w in line.lower() for w in ignore_words):
            if not scale_pattern.search(line) and not date_pattern.search(line):
                candidate_titles.append(line)

    if metadata["Title"] == "Not found" and candidate_titles:
        metadata["Title"] = candidate_titles[0]

    return metadata


def process_pil_image_ocr(img: Image.Image, max_dimension: int = 3500) -> str:
    """Schneidet das Schriftfeld zu, skaliert, führt Vorverarbeitung und OCR aus."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if FOCUS_BOTTOM_RIGHT:
        w, h = img.size
        img = img.crop((int(w * 0.50), int(h * 0.60), w, h))

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    processed_cv_img = preprocess_for_ocr(img)

    return pytesseract.image_to_string(
        processed_cv_img, lang="deu+eng", config=TESSERACT_CONFIG
    )


def extract_texts_from_file_v2(file_path: str, max_dimension: int = 3500) -> list[tuple[str, str]]:
    """Lädt Bilder oder PDFs und gibt eine Liste von (Seitenlabel, OCR-Text) zurück."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        results = []
        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)

        for page_index in range(total_pages):
            page = pdf[page_index]
            # 300 DPI Rendering für exakte Texterkennung
            bitmap = page.render(scale=300 / 72)
            pil_image = bitmap.to_pil()

            raw_text = process_pil_image_ocr(pil_image, max_dimension=max_dimension)
            label = f"Page {page_index + 1}/{total_pages}" if total_pages > 1 else ""
            results.append((label, raw_text))

        return results
    else:
        with Image.open(file_path) as img:
            raw_text = process_pil_image_ocr(img, max_dimension=max_dimension)
            return [("", raw_text)]


# ============================================================
# MAIN BATCH PROCESSING LOOP
# ============================================================

folder_path = input("Enter the path to the folder with files (images/PDFs): ").strip().strip('"\'')

if not os.path.isdir(folder_path):
    print(f"Error: Folder does not exist -> {folder_path}")
    sys.exit(1)

all_files = [
    f for f in sorted(os.listdir(folder_path))
    if f.lower().endswith(SUPPORTED_EXTENSIONS)
]

if not all_files:
    print(f"No compatible files found in: {folder_path}")
    sys.exit(0)

print(f"\nFound {len(all_files)} files. Starting application_ocr_v2...")
print(f"Results will be written to: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Processing: {filename}...")

        try:
            pages = extract_texts_from_file_v2(file_path)

            for page_label, raw_text in pages:
                sub_info = f" ({page_label})" if page_label else ""
                if page_label:
                    print(f"   -> OCR on {page_label}...")

                metadata = extract_metadata_v2(raw_text)

                result_text = (
                    f"Title: {metadata['Title']}\n"
                    f"Date: {metadata['Date']}\n"
                    f"ID: {metadata['ID']}\n"
                    f"Scale: {metadata['Scale']}"
                )

                entry = (
                    f"FILE: {filename}{sub_info}\n"
                    f"{'-' * 40}\n"
                    f"{result_text}\n"
                    f"{'=' * 60}\n\n"
                )

                out.write(entry)
                out.flush()

        except Exception as e:
            result_text = f"Error processing file: {str(e)}"
            print(f"   -> Failed: {e}")
            entry = (
                f"FILE: {filename}\n"
                f"{'-' * 40}\n"
                f"{result_text}\n"
                f"{'=' * 60}\n\n"
            )
            out.write(entry)
            out.flush()

        print("   -> Done.")

print(f"\nBatch processing finished successfully! Check '{output_file_path}'.")