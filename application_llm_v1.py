import base64
import io
import os
from pathlib import Path
import sys
from openai import OpenAI
from PIL import Image
import pypdfium2 as pdfium

# Verhindert Warnungen bei extrem großen Scans (z. B. TIFFs aus Archiven)
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# KONFIGURATION
# ============================================================

OUTPUT_FILE = "results_local_vlm.txt"
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
)

# WICHTIG: Planköpfe bei Bauplänen (DIN EN ISO 7200) liegen unten rechts.
# Durch den Crop erhält das 3B-Modell maximale Lesbarkeit ohne Detailverlust.
FOCUS_BOTTOM_RIGHT = True

# Lokale Ollama-Verbindung
LOCAL_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5vl:3b"

client = OpenAI(
    api_key="ollama",  # Beliebiger String, wird lokal nicht geprüft
    base_url=LOCAL_BASE_URL,
)

# ============================================================
# PROMPT DEFINITION
# ============================================================

PROMPT = """Analysiere dieses Schriftfeld/diesen Bauplan der Berliner Verkehrsbetriebe (BVG) und extrahiere die folgenden Metadaten:

1. Title: Der Haupttitel des Plans, Bauwerksname oder Planinhalt.
2. Date: Das Erstellungs-, Prüf- oder Revisionsdatum.
3. ID: Die Plannummer, Zeichnungsnummer, Blattnummer oder Dokumentencode.
4. Scale: Der Maßstab (z. B. 1:100, 1:50, 1:200/500).

Gib das Ergebnis strikt in folgendem Format aus (falls ein Feld nicht existiert, schreibe 'Not found'):
Title: <value>
Date: <value>
ID: <value>
Scale: <value>

Keine Einleitung oder Schlusssätze."""


def process_pil_image(img: Image.Image, max_dimension: int = 1536) -> str:
    """Schneidet bei Bedarf das Schriftfeld aus, skaliert moderat herunter
    und gibt einen kompakten Base64-JPEG-String zurück.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # 1. Plankopf-Ausschnitt (untere 35%, rechte 45%)
    if FOCUS_BOTTOM_RIGHT:
        w, h = img.size
        img = img.crop((int(w * 0.55), int(h * 0.65), w, h))

    # 2. Skalierung (1536px reicht für einen gecroppten Plankopf völlig aus)
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_file_as_base64_list(file_path: str, max_dimension: int = 1536) -> list[tuple[str, str]]:
    """Lädt Bild- oder PDF-Dateien und liefert eine Liste aus:
    (Seiten-Label, Base64-JPEG-String).
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        pages = []
        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)

        for page_index in range(total_pages):
            page = pdf[page_index]
            # 200 DPI Rendering (200 / 72 ≈ 2.77) für scharfe Details am Schriftkopf
            bitmap = page.render(scale=200 / 72)
            pil_image = bitmap.to_pil()
            b64_str = process_pil_image(pil_image, max_dimension=max_dimension)
            
            label = f"Page {page_index + 1}/{total_pages}" if total_pages > 1 else ""
            pages.append((label, b64_str))

        return pages
    else:
        with Image.open(file_path) as img:
            return [("", process_pil_image(img, max_dimension=max_dimension))]


# ============================================================
# BATCH-VERARBEITUNG
# ============================================================

folder_path = input("Pfad zum Ordner (Bilder/PDFs) eingeben: ").strip().strip('"\'')

if not os.path.isdir(folder_path):
    print(f"Fehler: Ordner nicht gefunden -> {folder_path}")
    sys.exit(1)

all_files = [
    f for f in sorted(os.listdir(folder_path))
    if f.lower().endswith(SUPPORTED_EXTENSIONS)
]

if not all_files:
    print(f"Keine passenden Dateien gefunden in: {folder_path}")
    sys.exit(0)

script_dir = Path(__file__).resolve().parent
output_file_path = script_dir / OUTPUT_FILE

print(f"\n{len(all_files)} Dateien gefunden. Starte lokale VLM-Verarbeitung ({MODEL_NAME})...")
print(f"Ergebnisse werden gespeichert in: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Verarbeite: {filename}...")

        try:
            pages = load_file_as_base64_list(file_path)

            for page_label, image_base64 in pages:
                sub_info = f" ({page_label})" if page_label else ""
                if page_label:
                    print(f"   -> Analysiere {page_label}...")

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=0.1,
                )

                result_text = response.choices[0].message.content.strip()

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
            print(f"   -> Fehler: {e}")
            entry = (
                f"FILE: {filename}\n"
                f"{'-' * 40}\n"
                f"{result_text}\n"
                f"{'=' * 60}\n\n"
            )
            out.write(entry)
            out.flush()

        print("   -> Fertig.")

print(f"\nVerarbeitung abgeschlossen! Ergebnisse in '{output_file_path}'.")