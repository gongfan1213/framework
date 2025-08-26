from paddleocr import PaddleOCR
from PIL import Image
import numpy as np


_ocr = PaddleOCR(lang="ch")


def ocr_image(img: Image.Image) -> str:
    arr = np.array(img)
    result = _ocr.ocr(arr, cls=True)
    texts = []
    for line in result:
        for _, (text, conf) in line:
            texts.append(text)
    return "\n".join(texts)


