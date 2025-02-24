import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import os

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    
    # Extract the token and the embeds from the loaded file
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # Cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # Add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    
    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token

def Distance_loss(images):
    # Calculate distances between RGB channels
    red = images[:,0]
    green = images[:,1]
    blue = images[:,2]
    
    # Calculate pairwise distances between channels
    rg_distance = torch.abs(red - green).mean()
    rb_distance = torch.abs(red - blue).mean()
    gb_distance = torch.abs(green - blue).mean()
    
    # Sum all distances to maximize color separation
    error = rg_distance + rb_distance + gb_distance
    return error

def apply_loss_and_generate(pipe, prompt, num_inference_steps=50, guidance_scale=7.5):
    # First generate without loss
    with autocast("cuda"):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    
    # Now generate with loss guidance
    def callback_fn(i, t, latents):
        if i % 5 == 0:  # Apply loss every 5 steps
            loss = Distance_loss(latents)
            latents = latents - 0.1 * torch.autograd.grad(loss, latents)[0]
        return latents
    
    with autocast("cuda"):
        image_with_loss = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            callback=callback_fn
        ).images[0]
    
    return image, image_with_loss

def main():
    # Initialize the pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # List of style embeddings to download (example styles)
    styles = [
        "sd-concepts-library/anime-lineart",
        "sd-concepts-library/watercolor-art",
        "sd-concepts-library/pixel-art",
        "sd-concepts-library/oil-painting",
        "sd-concepts-library/comic-style"
    ]

    # Load each style embedding
    style_tokens = []
    for style in styles:
        embed_path = hf_hub_download(repo_id=style, filename="learned_embeds.bin")
        token = load_learned_embed_in_clip(embed_path, pipe.text_encoder, pipe.tokenizer)
        style_tokens.append(token)

    # Get user prompt
    user_prompt = input("Enter your prompt: ")

    # Generate images for each style
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    
    for idx, (style_token, ax_row) in enumerate(zip(style_tokens, axes)):
        prompt = f"{user_prompt}, {style_token}"
        
        # Generate images with and without loss
        image_normal, image_with_loss = apply_loss_and_generate(pipe, prompt)
        
        # Display images
        ax_row[0].imshow(image_normal)
        ax_row[0].set_title(f"Style: {style_token} (Without Loss)")
        ax_row[0].axis('off')
        
        ax_row[1].imshow(image_with_loss)
        ax_row[1].set_title(f"Style: {style_token} (With Loss)")
        ax_row[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 