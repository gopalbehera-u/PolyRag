from functools import lru_cache
from pathlib import Path
from typing import Optional


from pydantic_settings import BaseSettings,SettingsConfigDict



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)
    # LLM
    gemini_api_key : str=""
    gemini_model :str ='gemini-2.5-flash'

    # Storage
    upload_dir : Path =Path('data/uploads')
    chroma_persist_dir : Path =Path('data/chroma')
    max_upload_mb : int=200


    # Chunking
    chunk_size : int=1000
    chunk_overlap : int =150

    # OCR/PDF
    poppler_path : Optional[str]=None
    ocr_languages : str='en'

    # EMbedding
    embeding_model_name : str = "all-MiniLM-L6-v2"

    # Retrival 
    default_top_k : int= 4

    def ensure_dir(self) -> None:
        self.upload_dir.mkdir(parents=True,exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True,exist_ok=True)




@lru_cache
def get_settings() -> Settings:
    settings=Settings()
    settings.ensure_dir()
    return settings