from functools import lru_cache

from config import get_settings

from utils.logger import get_logger


logger=get_logger(__name__)


@lru_cache

def _get_reader():
    import easyocr

    settings=get_settings()

    langs=[l.strip() for l in settings.ocr_languages.split(",") if l.strip()]
    logger.info("Loading EasyOCR reader for languages=%s...",langs)
    return easyocr.Reader(langs,gpu=False)



def extract_text(image)-> str:
    """`image` may be a file path (str) ir a numpy  ndarray."""
    reader=_get_reader()
    result=reader.readtext(image,detail=0)
    return "\n".join(result)