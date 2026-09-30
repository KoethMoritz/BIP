import base64
import io
import os
from pathlib import Path
import re  # NEU: Zum Bereinigen eventueller <think>-Tags
import sys
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import pypdfium2 as pdfium

# Prevent DecompressionBombWarning for very large scanned files
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

OUTPUT_FILE = "results_api.txt"
SUPPORTED_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
)

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("HTW_API_KEY")
BASE_URL = os.getenv("HTW_BASE_URL")
MODEL = os.getenv("MODEL_NAME")

# Validation
if not API_KEY:
    raise ValueError(f"HTW_API_KEY was not found in: {env_path}")
if not BASE_URL:
    raise ValueError(f"HTW_BASE_URL was not found in: {env_path}")

# ============================================================
# CONNECT TO HTW API
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ============================================================
# PROMPT DEFINITION
# ============================================================

PROMPT = """Analyze this document/image (especially title blocks, headers, stamps, or labels) and extract the following metadata:

1. Title: The main title, drawing title, or document heading.
2. Date: Any creation, issue, or revision date found.
3. ID: Any identification number, sheet/drawing number, document code, or registration number.
4. Scale: Any drawing scale or ratio (e.g., 1:100, 1:50, 1/4" = 1'-0", or bar scale).

Output the result strictly in the following format (if a field cannot be found, write 'Not found'):
Title: <value>
Date: <value>
ID: <value>
Scale: <value>

Do not include any conversational opening or closing text."""


def pil_image_to_base64(img: Image.Image, max_dimension: int = 2048) -> str:
    """Scales down a PIL Image in-memory and returns a compact Base64 JPEG string."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_file_as_base64_list(file_path: str, max_dimension: int = 2048) -> list[tuple[str, str]]:
    """Loads an image or PDF. 
    
    Returns a list of tuples: (page_label, base64_jpeg_string).
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        images = []
        pdf = pdfium.PdfDocument(file_path)
        total_pages = len(pdf)

        for page_index in range(total_pages):
            page = pdf[page_index]
            # Render page at 200 DPI (scale ~2.77) for good readability of title blocks
            bitmap = page.render(scale=200 / 72)
            pil_image = bitmap.to_pil()
            b64_str = pil_image_to_base64(pil_image, max_dimension=max_dimension)
            
            label = f"Page {page_index + 1}/{total_pages}" if total_pages > 1 else ""
            images.append((label, b64_str))
            
        return images
    else:
        # Standard image files
        with Image.open(file_path) as img:
            return [("", pil_image_to_base64(img, max_dimension=max_dimension))]


# ============================================================
# MAIN BATCH PROCESSING LOOP
# ============================================================

folder_path = input("Enter the path to the folder with files (images/PDF): ").strip().strip('"\'')

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

output_file_path = script_dir / OUTPUT_FILE

print(f"\nFound {len(all_files)} files. Starting batch processing...")
print(f"Results will be written to: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Processing: {filename}...")

        try:
            pages = load_file_as_base64_list(file_path)

            for page_label, image_base64 in pages:
                sub_info = f" ({page_label})" if page_label else ""
                if page_label:
                    print(f"   -> Analyzing {page_label} with reasoning/thinking...")

                # ============================================================
                # HIER ERFOLGT DIE AKTIVIERUNG DES THINKING-MODUS
                # ============================================================
                response = client.chat.completions.create(
                    model=MODEL,
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
                    # 1. Empfohlene Temperatur für Reasoning (0.6 - 1.0)
                    temperature=0.6,
                    # 2. Ausreichend Token-Budget für Gedankengang + Antwort
                    max_tokens=4096,
                    # 3. Parameter für vLLM / SGLang / OpenAI-kompatible Server
                    extra_body={
                        "enable_thinking": True,
                        "reasoning_effort": "high",  # "low", "medium", "high" oder "xhigh"
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                )

                choice_message = response.choices[0].message
                raw_text = choice_message.content or ""

                # Optional: Gedanken auf der Konsole ausgeben, falls der Server sie separat liefert
                reasoning = getattr(choice_message, "reasoning_content", None)
                if reasoning:
                    print(f"      [Thinking abgeschlossen: ~{len(reasoning.split())} Wörter]")

                # 4. Falls der Server <think>...</think> im Fließtext liefert, filtern wir es hier heraus:
                result_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

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

        print(f"   -> Done.")

print(f"\nBatch processing finished successfully! Check '{output_file_path}'.")