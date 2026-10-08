from pathlib import Path
import shutil

import fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from astra_model.ingestion import extract_file


def make_test_image(path: Path, text: str):
    image = Image.new("RGB", (1200, 240), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 64)
    draw.text((30, 60), text, fill="black", font=font)
    image.save(path)


def test_extracts_native_pdf_text(tmp_path):
    path = tmp_path / "digital.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Astra Nexus retrieves grounded information from documents.",
    )
    pdf.save(path)
    pdf.close()

    records = extract_file(path)

    assert len(records) == 1
    assert "Astra Nexus" in records[0]["text"]
    assert records[0]["page_number"] == 1
    assert records[0]["extraction_method"] == "native_pdf_text"


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
def test_extracts_text_from_image(tmp_path):
    path = tmp_path / "sample.png"
    make_test_image(path, "ASTRA NEXUS OCR TEST")

    records = extract_file(path)

    assert len(records) == 1
    assert records[0]["content_type"] == "image"
    assert "ASTRA" in records[0]["text"].upper()


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract not installed")
def test_ocr_fallback_for_scanned_pdf(tmp_path):
    image_path = tmp_path / "scan.png"
    make_test_image(image_path, "SCANNED DOCUMENT ASTRA")

    pdf_path = tmp_path / "scanned.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=600, height=120)
    page.insert_image(page.rect, filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()

    records = extract_file(pdf_path)

    assert len(records) == 1
    assert records[0]["extraction_method"] == "ocr"
    assert "SCANNED" in records[0]["text"].upper()


def test_rejects_unsupported_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("unsupported")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_file(path)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_file(tmp_path / "missing.pdf")
