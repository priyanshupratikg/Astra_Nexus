from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _ocr(image):
    text = pytesseract.image_to_string(image).strip()
    data = pytesseract.image_to_data(image, output_type=Output.DICT)

    scores = []
    for value in data["conf"]:
        try:
            score = float(value)
            if score >= 0:
                scores.append(score)
        except (ValueError, TypeError):
            pass

    confidence = round(sum(scores) / len(scores), 2) if scores else None
    return text, confidence


def extract_file(file_path, ocr_min_native_chars=30, render_zoom=2.0):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension in SUPPORTED_IMAGES:
        with Image.open(path) as image:
            text, confidence = _ocr(image.convert("RGB"))

        return [{
            "text": text,
            "source": path.name,
            "page_number": None,
            "content_type": "image",
            "extraction_method": "ocr",
            "ocr_confidence": confidence,
            "metadata": {"source_path": str(path)},
        }]

    if extension != ".pdf":
        raise ValueError(f"Unsupported file type: {extension}")

    results = []

    with fitz.open(path) as pdf:
        page_count = len(pdf)

        for index, page in enumerate(pdf):
            native_text = page.get_text("text").strip()
            confidence = None

            if len(native_text) >= ocr_min_native_chars:
                text = native_text
                method = "native_pdf_text"
            else:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(render_zoom, render_zoom),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                text, confidence = _ocr(image)

                if text:
                    method = "ocr"
                elif native_text:
                    text = native_text
                    method = "native_pdf_text_fallback"
                else:
                    method = "ocr"

            results.append({
                "text": text,
                "source": path.name,
                "page_number": index + 1,
                "content_type": "pdf_page",
                "extraction_method": method,
                "ocr_confidence": confidence,
                "metadata": {
                    "source_path": str(path),
                    "page_count": page_count,
                },
            })

    return results
