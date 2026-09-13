from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


from config import get_settings

from utils.logger import get_logger


logger=get_logger(__name__)


@lru_cache
def get_embedding_model() -> HuggingFaceEmbeddings:
    settings=get_settings()
    logger.info("Loading embeding model '%s'...",settings.embeding_model_name)
    model=HuggingFaceEmbeddings(model_name=settings.embeding_model_name)
    logger.info("Embedding model loaded.")
    return model