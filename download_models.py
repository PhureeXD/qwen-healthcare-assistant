import os
import time
from huggingface_hub import snapshot_download, hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder

# Define models to download
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

def download_with_retries(repo_id, retries=5, delay=10):
    """Downloads a model with retry logic."""
    for i in range(retries):
        try:
            print(f"Downloading {repo_id} (Attempt {i+1}/{retries})...")
            # resume_download=True ensures we don't start from scratch if interrupted
            snapshot_download(repo_id=repo_id, resume_download=True)
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
    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    download_with_retries(EMBEDDING_MODEL_NAME)
    
    # Also initialize SentenceTransformer to ensure it caches correctly for the library
    print(f"Initializing SentenceTransformer for {EMBEDDING_MODEL_NAME} to populate cache...")
    try:
        SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Warning: Failed to initialize SentenceTransformer: {e}")

    print(f"Downloading cross-encoder model: {CROSS_ENCODER_MODEL_NAME}")
    download_with_retries(CROSS_ENCODER_MODEL_NAME)
    
    # Initialize CrossEncoder to populate cache
    print(f"Initializing CrossEncoder for {CROSS_ENCODER_MODEL_NAME} to populate cache...")
    try:
        CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    except Exception as e:
        print(f"Warning: Failed to initialize CrossEncoder: {e}")

    # Download GGUF model
    llm_repo_id = "phureexd/qwen3_v2_gguf"
    llm_filename = "unsloth.Q4_K_M.gguf"
    print(f"Downloading LLM: {llm_filename} from {llm_repo_id}")
    try:
        hf_hub_download(repo_id=llm_repo_id, filename=llm_filename, local_dir=".", local_dir_use_symlinks=False)
        print(f"Successfully downloaded {llm_filename}")
    except Exception as e:
        print(f"Error downloading LLM: {e}")
        raise e

    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
