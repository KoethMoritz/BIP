import os
import platform
import re
import shutil
import sys
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
import pytesseract

# Prevent DecompressionBombWarning for very large scanned files
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

OUTPUT_FILE = "results_ocr_v1.txt"
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
)

# Bauplan-Schriftfelder liegen standardmäßig unten rechts.
# Schneidet die unteren 35% und rechten 45% aus -> spart massiv Zeit bei A0/A1 Plänen.
# Auf False setzen, falls Schriftfelder bei deinen Plänen woanders liegen.
FOCUS_BOTTOM_RIGHT = True

# ============================================================
# TESSERACT BINARY PATH DETECTION (Mac & Windows)
# ============================================================

if platform.system() == "Darwin":  # macOS (M1/M2/Intel)
    system_tesseract = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    if os.path.exists(system_tesseract):
        pytesseract.pytesseract.tesseract_cmd = system_tesseract
    elif os.path.exists("/usr/local/bin/tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

elif platform.system() == "Windows":  # Windows
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in win_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Skript-Ordner zur Speicherung der Ergebnisse
script_dir = Path(__file__).resolve().parent
output_file_path = script_dir / OUTPUT_FILE


# ============================================================
# METADATA EXTRACTION LOGIC
# ============================================================

def extract_metadata(raw_text: str) -> dict:
    """Extrahiert Title, Date, ID und Scale über RegEx und Heuristiken."""
    metadata = {
        "Title": "Not found",
        "Date": "Not found",
        "ID": "Not found",
        "Scale": "Not found",
    }

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 1. Scale / Maßstab (z. B. 1:100, 1:50, 1/4" = 1'-0")
    scale_pattern = re.compile(
        r"(?:M(?:aßstab)?\s*[:\.]?\s*)?(1\s*:\s*\d{1,4}|1/\d+[\"']?\s*=\s*\d+)",
        re.IGNORECASE,
    )

    # 2. Date / Datum (z. B. 24.11.2023, 2023-11-24, 24/11/2023)
    date_pattern = re.compile(
        r"\b(\d{1,2}[\.\/\-]\d{1,2}[\.\/\-]\d{2,4}|\d{4}[\.\/\-]\d{2}[\.\/\-]\d{2})\b"
    )

    # 3. ID / Plan-Nr. Keywords
    id_pattern = re.compile(
        r"(?:plan(?:-|\s*)?nr\.?|zeichnungs(?:-|\s*)?nr\.?|dok(?:ument)?(?:-|\s*)?nr\.?|id|code|nr\.)\s*[:\.]?\s*([A-Z0-9\-_./]+)",
        re.IGNORECASE,
    )

    # 4. Title Keywords
    title_pattern = re.compile(
        r"(?:titel|bezeichnung|bauvorhaben|planinhalt|projekt)\s*[:\.]?\s*(.*)",
        re.IGNORECASE,
    )

    candidate_titles = []

    for i, line in enumerate(lines):
        clean = line.strip()

        # Scale
        if metadata["Scale"] == "Not found":
            m_scale = scale_pattern.search(clean)
            if m_scale:
                metadata["Scale"] = m_scale.group(1).replace(" ", "")

        # Date
        if metadata["Date"] == "Not found":
            m_date = date_pattern.search(clean)
            if m_date:
                metadata["Date"] = m_date.group(1)

        # ID
        if metadata["ID"] == "Not found":
            m_id = id_pattern.search(clean)
            if m_id and m_id.group(1):
                metadata["ID"] = m_id.group(1).strip()
            elif (
                any(k in clean.lower() for k in ["plan-nr", "plannr", "zeichnungsnr"])
                and i + 1 < len(lines)
            ):
                next_val = lines[i + 1].strip()
                if len(next_val) < 40 and not any(
                    s in next_val.lower() for s in ["maßstab", "datum"]
                ):
                    metadata["ID"] = next_val

        # Title
        if metadata["Title"] == "Not found":
            m_title = title_pattern.search(clean)
            if m_title and len(m_title.group(1).strip()) > 3:
                metadata["Title"] = m_title.group(1).strip()

        # Sammle Zeilen für Fallback-Titel
        if (
            len(clean) > 5
            and not scale_pattern.search(clean)
            and not date_pattern.search(clean)
        ):
            candidate_titles.append(clean)

    # Fallback-Titel: Erste signifikante Textzeile
    if metadata["Title"] == "Not found" and candidate_titles:
        metadata["Title"] = candidate_titles[0]

    return metadata


def process_pil_image_ocr(img: Image.Image, max_dimension: int = 3000) -> str:
    """Schneidet bei Bedarf den Plankopf aus, skaliert und führt OCR aus."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Plankopf-Fokus (unten rechts)
    if FOCUS_BOTTOM_RIGHT:
        w, h = img.size
        img = img.crop((int(w * 0.55), int(h * 0.65), w, h))

    # Skalieren auf max. Dimension zur Performance-Optimierung
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    # Tesseract-OCR mit Deutsch und Englisch ausführen
    return pytesseract.image_to_string(img, lang="deu+eng")


def extract_texts_from_file(file_path: str, max_dimension: int = 3000) -> list[tuple[str, str]]:
    """Lädt Bild- oder PDF-Dateien und liefert eine Liste aus:
    (Seiten-Label, OCR-Rohtext).
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        results = []
        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)

        for page_index in range(total_pages):
            page = pdf[page_index]
            # 300 DPI Rendering (300 / 72 ≈ 4.166) liefert optimale Schärfe für OCR
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

print(f"\nFound {len(all_files)} files. Starting local OCR processing...")
print(f"Results will be written to: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Processing: {filename}...")

        try:
            page_results = extract_texts_from_file(file_path)

            for page_label, raw_text in page_results:
                sub_info = f" ({page_label})" if page_label else ""
                if page_label:
                    print(f"   -> OCR on {page_label}...")

                metadata = extract_metadata(raw_text)

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