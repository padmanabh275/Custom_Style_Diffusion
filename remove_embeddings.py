import shutil
from pathlib import Path

def remove_embeddings_folder():
    embed_dir = Path("embeddings")
    if embed_dir.exists():
        try:
            shutil.rmtree(embed_dir)
            print("✓ Successfully removed embeddings folder")
        except Exception as e:
            print(f"✗ Error removing embeddings folder: {e}")
    else:
        print("Embeddings folder does not exist")

if __name__ == "__main__":
    remove_embeddings_folder() 