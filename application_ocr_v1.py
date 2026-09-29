import os
import platform
import re
import shutil
import sys
from pathlib import Path
from PIL import Image
import pytesseract

# Prevent DecompressionBombWarning for very large scanned files
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

OUTPUT_FILE = "results_ocr_v1.txt"
SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

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


def process_image(file_path: str, max_dimension: int = 3000) -> str:
    """Lädt das Bild, schneidet bei Bedarf den Plankopf aus und führt OCR aus."""
    with Image.open(file_path) as img:
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
        text = pytesseract.image_to_string(img, lang="deu+eng")
        return text


# ============================================================
# MAIN BATCH PROCESSING LOOP
# ============================================================

folder_path = input("Enter the path to the folder with images: ").strip().strip('"\'')

if not os.path.isdir(folder_path):
    print(f"Error: Folder does not exist -> {folder_path}")
    sys.exit(1)

all_files = [
    f for f in sorted(os.listdir(folder_path))
    if f.lower().endswith(SUPPORTED_EXTENSIONS)
]

if not all_files:
    print(f"No compatible images found in: {folder_path}")
    sys.exit(0)

print(f"\nFound {len(all_files)} images. Starting local OCR processing...")
print(f"Results will be written to: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Processing: {filename}...")

        try:
            raw_text = process_image(file_path)
            metadata = extract_metadata(raw_text)

            result_text = (
                f"Title: {metadata['Title']}\n"
                f"Date: {metadata['Date']}\n"
                f"ID: {metadata['ID']}\n"
                f"Scale: {metadata['Scale']}"
            )
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