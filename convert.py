import os
import fitz
import subprocess
from PIL import Image
from music21 import converter
from utils import MSCORE


def abc2xml(abc_content, output_xml_path):
    score = converter.parse(abc_content, format="abc")
    score.write("musicxml", fp=output_xml_path, encoding="utf-8")
    return output_xml_path


def xml2(xml_path: str, target_fmt: str):
    src_fmt = os.path.basename(xml_path).split(".")[-1]
    if not "." in target_fmt:
        target_fmt = "." + target_fmt

    target_file = xml_path.replace(f".{src_fmt}", target_fmt)
    command = [MSCORE, "-o", target_file, xml_path]
    result = subprocess.run(command)
    print(result)
    return target_file


def pdf2img(pdf_path: str):
    output_path = pdf_path.replace(".pdf", ".jpg")
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(doc.page_count):
        page = doc[page_number]
        image = page.get_pixmap()
        images.append(
            Image.frombytes("RGB", [image.width, image.height], image.samples)
        )

    merged_image = Image.new(
        "RGB", (images[0].width, sum(image.height for image in images))
    )
    y_offset = 0
    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height

    merged_image.save(output_path, "JPEG")
    doc.close()
    return output_path


def xml2img(xml_file: str):
    ext = os.path.basename(xml_file).split(".")[-1]
    pdf_score = xml_file.replace(f".{ext}", ".pdf")
    command = [MSCORE, "-o", pdf_score, xml_file]
    result = subprocess.run(command)
    print(result)
    return pdf_score, pdf2img(pdf_score)
