import base64
import io
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Prevent DecompressionBombWarning for very large scanned files
Image.MAX_IMAGE_PIXELS = None

# ============================================================
# CONFIGURATION & SETTINGS
# ============================================================

OUTPUT_FILE = "results_api.txt"
SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Ensures .env is found in the same folder as this script, regardless of where terminal runs
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


def process_image_to_base64(file_path: str, max_dimension: int = 2048) -> str:
    """Opens any image format (including multi-MB TIFFs), scales it down in-memory,

    and returns a compact Base64 JPEG string.
    """
    with Image.open(file_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


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

# Store results in the script's directory
output_file_path = script_dir / OUTPUT_FILE

print(f"\nFound {len(all_files)} images. Starting batch processing...")
print(f"Results will be written to: {output_file_path}\n")

with open(output_file_path, "a", encoding="utf-8") as out:
    for index, filename in enumerate(all_files, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{index}/{len(all_files)}] Processing: {filename}...")

        try:
            image_base64 = process_image_to_base64(file_path)

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
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()

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