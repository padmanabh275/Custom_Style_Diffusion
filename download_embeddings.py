import os
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_embeddings():
    # Create embeddings directory
    embed_dir = Path("embeddings")
    embed_dir.mkdir(exist_ok=True)
    
    # Updated list of specific style concepts
    styles = [
        "sd-concepts-library/ronaldo",
        "sd-concepts-library/canna-lily-flowers102",
        "sd-concepts-library/threestooges",
        "sd-concepts-library/pop-art",
        "sd-concepts-library/bird-style"
    ]
    
    # Download each embedding
    for style in styles:
        style_name = style.split('/')[-1]
        print(f"Downloading {style_name}...")
        
        try:
            # Download the embedding
            embed_path = hf_hub_download(
                repo_id=style,
                filename="learned_embeds.bin",
                local_dir=embed_dir / style_name
            )
            print(f"✓ Successfully downloaded to {embed_path}")
            
        except Exception as e:
            print(f"✗ Error downloading {style_name}: {e}")
            print("Try visiting: https://huggingface.co/" + style)
    
    print("\nDownload complete! Embeddings are stored in the 'embeddings' directory.")

if __name__ == "__main__":
    download_embeddings() 