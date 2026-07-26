import os
import re
import csv
import glob
import base64
import time
import io
import sys
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# =============================================================
# KONFIGURASI
# =============================================================
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen2-vl-2b-instruct"   # from your log

DATASET_ROOT = "./Indonesian License Plate Dataset"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "test")
LABELS_LP_DIR = os.path.join(DATASET_ROOT, "labelswithLP", "test")

OUTPUT_CSV = "johasil.csv"
OUTPUT_SUMMARY = "josummary.txt"

CROP_PADDING_RATIO = 0.12
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Improved prompt – gives a concrete example
PROMPT = (
    "Read the Indonesian license plate number in this image. "
    "Ignore any small text at the bottom (like expiration dates). "
    "The plate format is like 'B 1234 XYZ' – output only the alphanumeric characters, "
    "e.g., 'B1234XYZ'. Do not add any explanation."
)

FALLBACK_PROMPT = (
    "What is the license plate number? Respond with only the letters and digits, "
    "e.g., 'B1234XYZ'."
)

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

# =============================================================
# CEK KONEKSI
# =============================================================
def check_lmstudio_connection():
    base_url = LMSTUDIO_URL.replace("/v1/chat/completions", "/v1/models")
    try:
        resp = requests.get(base_url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        available_models = [m.get("id", "") for m in data.get("data", [])]
        print(f"[INFO] Terhubung ke LM Studio. Model tersedia: {available_models}")
        if MODEL_NAME not in available_models:
            print(f"[WARNING] MODEL_NAME='{MODEL_NAME}' tidak terdaftar.")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal koneksi ke LM Studio: {e}")
        return False

# =============================================================
# PARSING LABEL
# =============================================================
def find_image_path(images_dir: str, stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
        p_upper = os.path.join(images_dir, stem + ext.upper())
        if os.path.isfile(p_upper):
            return p_upper
    return None

def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    try:
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
    except ValueError:
        return None
    plate_text = " ".join(parts[5:]).strip().upper()
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "ground_truth": plate_text,
    }

# =============================================================
# CROP & ENHANCE
# =============================================================
def crop_plate(image: Image.Image, box: dict) -> Image.Image:
    img_w, img_h = image.size
    xc, yc, w, h = box["x_center"], box["y_center"], box["width"], box["height"]

    w_pad = w * (1 + CROP_PADDING_RATIO)
    h_pad = h * (1 + CROP_PADDING_RATIO)

    xmin = (xc - w_pad / 2) * img_w
    xmax = (xc + w_pad / 2) * img_w
    ymin = (yc - h_pad / 2) * img_h
    ymax = (yc + h_pad / 2) * img_h

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(img_w, int(xmax))
    ymax = min(img_h, int(ymax))

    return image.crop((xmin, ymin, xmax, ymax))

def enhance_plate_image(crop: Image.Image) -> Image.Image:
    # Convert to grayscale for better contrast?
    # But the VLM expects RGB; keep RGB but enhance
    enhancer = ImageEnhance.Contrast(crop)
    crop = enhancer.enhance(1.8)
    enhancer = ImageEnhance.Sharpness(crop)
    crop = enhancer.enhance(2.0)
    # Optional: auto contrast
    crop = ImageOps.autocontrast(crop, cutoff=2)
    return crop

def image_to_base64(image: Image.Image) -> str:
    # Resize to a reasonable size to reduce token usage and standardise input
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =============================================================
# PANGGIL LM STUDIO
# =============================================================
def query_lmstudio(crop_image: Image.Image, use_fallback: bool = False) -> str:
    prompt = FALLBACK_PROMPT if use_fallback else PROMPT
    b64_img = image_to_base64(crop_image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 50,
    }

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            print(f"    [RAW OUTPUT] {raw_text}")
            return raw_text.strip()
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    print(f"[WARNING] Query gagal: {last_err}")
    return ""

# =============================================================
# POST-PROCESSING YANG LEBIH CERDAS
# =============================================================
def correct_common_errors(text: str) -> str:
    # Remove spaces, hyphens, etc.
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text

def extract_plate_number(text: str) -> str:
    """
    Extract the plate number using a multi‑step approach.
    Returns the most plausible plate string, or empty string if none found.
    """
    raw = correct_common_errors(text)
    if not raw:
        return ""

    # Pattern for Indonesian plates: 1-2 letters, 1-4 digits, 0-3 letters
    pattern = re.compile(r'([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})')

    # Step 1: try direct match
    m = pattern.search(raw)
    if m:
        return m.group(0)

    # Step 2: try swapping common confusions (digit<->letter)
    # We'll try each position and see if any swap yields a match
    confusion = {
        '8': 'B', 'B': '8',
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1',
        '5': 'S', 'S': '5',
        '2': 'Z', 'Z': '2'
    }

    # Try up to two swaps
    n = len(raw)
    # For each position, try swapping that character
    for i in range(n):
        ch = raw[i]
        if ch in confusion:
            swapped = list(raw)
            swapped[i] = confusion[ch]
            candidate = ''.join(swapped)
            m = pattern.search(candidate)
            if m:
                return m.group(0)

    # Step 3: if the string starts with digits and has letters later, try to move
    # the first digit to the front as a letter if it matches a known letter.
    # For example "82136PZJ" -> we try "B2136PZJ" by replacing first char.
    # We'll try all possible first-character replacements that are plausible.
    # But we need to know which letter is likely. We can try common ones: B, D, F, etc.
    # Instead, we can try to see if any prefix of digits can be turned into a letter.
    # For "82136PZJ", if we replace '8' with 'B', we get "B2136PZJ" which matches pattern? "B2136PZJ" -> letters: B, digits: 2136, suffix: PZJ -> ok.
    # So we can try to replace the first digit with its letter equivalent.
    # We'll iterate over all possible swaps in the prefix that consists of digits.
    # Find where the first letter occurs.
    first_letter_pos = None
    for i, ch in enumerate(raw):
        if ch.isalpha():
            first_letter_pos = i
            break
    if first_letter_pos is not None and first_letter_pos > 0:
        # There are digits before the first letter
        prefix = raw[:first_letter_pos]
        suffix = raw[first_letter_pos:]
        # Try to convert the entire prefix into letters? Not reliable.
        # Instead, try to take the prefix, and for each character, replace with a letter if possible.
        # But we can try to replace the first digit only.
        ch = prefix[0]
        if ch in confusion:
            # replace with its letter counterpart
            new_prefix = confusion[ch] + prefix[1:] if len(prefix) > 1 else confusion[ch]
            candidate = new_prefix + suffix
            m = pattern.search(candidate)
            if m:
                return m.group(0)

    # Step 4: fallback – return the cleaned raw text (may be incorrect but at least something)
    return raw

def get_prediction(crop_image: Image.Image) -> str:
    # Primary attempt
    pred = query_lmstudio(crop_image, use_fallback=False)
    plate = extract_plate_number(pred)
    # If result seems too short (less than 5 characters), try fallback prompt
    if len(plate) < 5:
        pred_fb = query_lmstudio(crop_image, use_fallback=True)
        plate = extract_plate_number(pred_fb)
    return plate

# =============================================================
# CER COMPUTATION
# =============================================================
def levenshtein_distance(ref: str, hyp: str):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def compute_cer(ground_truth: str, prediction: str) -> float:
    ref = ground_truth.replace(" ", "")
    hyp = prediction.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = levenshtein_distance(ref, hyp)
    return round(dist / len(ref), 4)

# =============================================================
# MAIN
# =============================================================
def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_LP_DIR):
        raise FileNotFoundError(f"Folder label tidak ditemukan: {LABELS_LP_DIR}")

    if not check_lmstudio_connection():
        print("[ERROR] Program dihentikan.")
        sys.exit(1)

    label_files = sorted(glob.glob(os.path.join(LABELS_LP_DIR, "*.txt")))
    if not label_files:
        raise FileNotFoundError(f"Tidak ada file label di {LABELS_LP_DIR}")

    print(f"[INFO] Ditemukan {len(label_files)} file label")

    total_cer = 0.0
    total_plates = 0
    exact_match_count = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()

        for lf_idx, label_path in enumerate(label_files, 1):
            stem = os.path.splitext(os.path.basename(label_path))[0]
            image_path = find_image_path(IMAGES_DIR, stem)

            if image_path is None:
                print(f"[WARNING] Gambar untuk {stem} tidak ditemukan, dilewati.")
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]

            boxes = [parse_label_line(l) for l in lines]
            boxes = [b for b in boxes if b is not None]

            if not boxes:
                continue

            image = Image.open(image_path).convert("RGB")
            print(f"[{lf_idx}/{len(label_files)}] {stem}: {len(boxes)} plat terdeteksi")

            for plate_idx, box in enumerate(boxes):
                ground_truth = box["ground_truth"]
                crop = crop_plate(image, box)
                crop_enhanced = enhance_plate_image(crop)

                prediction = get_prediction(crop_enhanced)
                cer_score = compute_cer(ground_truth, prediction)

                total_cer += cer_score
                total_plates += 1
                if prediction == ground_truth.replace(" ", ""):
                    exact_match_count += 1

                image_label = f"{stem}_{plate_idx}"
                print(f"    [{image_label}] GT: '{ground_truth}' | Pred: '{prediction}' | CER: {cer_score}")

                writer.writerow({
                    "image": image_label,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "CER_score": cer_score,
                })

    avg_cer = round(total_cer / total_plates, 4) if total_plates else 0.0
    accuracy = round(exact_match_count / total_plates, 4) if total_plates else 0.0

    summary_lines = [
        f"Total plat diproses : {total_plates}",
        f"Rata-rata CER       : {avg_cer}",
        f"Exact match         : {exact_match_count}/{total_plates}",
        f"Accuracy (exact)    : {accuracy}",
        f"Model               : {MODEL_NAME}",
    ]

    print("\n[SELESAI]")
    for line in summary_lines:
        print(f"[HASIL] {line}")
    print(f"[HASIL] CSV disimpan di {OUTPUT_CSV}")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[HASIL] Ringkasan disimpan di {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    main()