import cv2

import numpy as np

from services.ocr import extract_text
from utils.logger import get_logger

logger=get_logger(__name__)

def _preprocess(image_path:str) -> np.ndarray:
    image=cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read imagge file : {image_path}")

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    denoised=cv2.fastNlMeansDenoising(gray,h=10)
    thresh=cv2.adaptiveThreshold(
         denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh



def extract_text_from_image(image_path:str) -> str:
    processed=_preprocess(image_path)
    text=extract_text(processed)


    if not text.strip():
        logger.warning("OCR produced no text for image=%s",image_path)

    return text