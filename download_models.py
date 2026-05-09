import os
import time

from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

# Define models to download
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def get_hf_token():
    """Reads an optional Hugging Face token from the environment."""
    return os.getenv("HF_TOKEN") or None


def download_with_retries(repo_id, token=None, retries=5, delay=10):
    """Downloads a model with retry logic."""
    for i in range(retries):
        try:
            print(f"Downloading {repo_id} (Attempt {i+1}/{retries})...")
            snapshot_download(repo_id=repo_id, token=token)
            print(f"Successfully downloaded {repo_id}")
            return
        except Exception as e:
            print(f"Error downloading {repo_id}: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to download {repo_id} after {retries} attempts.")
                raise e


def download_models():
    hf_token = get_hf_token()
    if hf_token:
        print("Using HF_TOKEN for Hugging Face downloads.")
    else:
        print("HF_TOKEN is not set. Hugging Face downloads will be unauthenticated.")

    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    download_with_retries(EMBEDDING_MODEL_NAME, token=hf_token)

    # Also initialize SentenceTransformer to ensure it caches correctly for the library
    print(
        f"Initializing SentenceTransformer for {EMBEDDING_MODEL_NAME} to populate cache..."
    )
    try:
        SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Warning: Failed to initialize SentenceTransformer: {e}")

    print(f"Downloading cross-encoder model: {CROSS_ENCODER_MODEL_NAME}")
    download_with_retries(CROSS_ENCODER_MODEL_NAME, token=hf_token)

    # Initialize CrossEncoder to populate cache
    print(
        f"Initializing CrossEncoder for {CROSS_ENCODER_MODEL_NAME} to populate cache..."
    )
    try:
        CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    except Exception as e:
        print(f"Warning: Failed to initialize CrossEncoder: {e}")

    # Download GGUF model
    llm_repo_id = "phureexd/qwen35_medical_lora_gguf"
    llm_filename = "Qwen3.5-2B.Q4_K_M.gguf"

    print(f"Downloading LLM: {llm_filename} from {llm_repo_id}")
    try:
        hf_hub_download(
            repo_id=llm_repo_id,
            filename=llm_filename,
            local_dir=".",
            token=hf_token,
        )
        print(f"Successfully downloaded {llm_filename}")
    except Exception as e:
        print(f"Error downloading LLM: {e}")
        raise e

    print("All models downloaded successfully.")


if __name__ == "__main__":
    download_models()
