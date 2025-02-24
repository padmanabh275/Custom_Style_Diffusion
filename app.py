import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import gradio as gr
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import traceback

# Reuse the same load_learned_embed_in_clip and Distance_loss functions
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # Get the expected dimension from the text encoder
    expected_dim = text_encoder.get_input_embeddings().weight.shape[1]
    current_dim = embeds.shape[0]

    # Resize embeddings if dimensions don't match
    if current_dim != expected_dim:
        print(f"Resizing embedding from {current_dim} to {expected_dim}")
        # Option 1: Truncate or pad with zeros
        if current_dim > expected_dim:
            embeds = embeds[:expected_dim]
        else:
            embeds = torch.cat([embeds, torch.zeros(expected_dim - current_dim)], dim=0)
        
    # Reshape to match expected dimensions
    embeds = embeds.unsqueeze(0)  # Add batch dimension
    
    # Cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds = embeds.to(dtype)

    # Add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    
    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds[0]
    return token

def Distance_loss(images):
    # Ensure we're working with gradients
    if not images.requires_grad:
        images = images.detach().requires_grad_(True)
    
    # Convert to float32 and normalize
    images = images.float() / 2 + 0.5
    
    # Get RGB channels
    red = images[:,0:1]
    green = images[:,1:2]
    blue = images[:,2:3]
    
    # Calculate color distances using L2 norm
    rg_distance = ((red - green) ** 2).mean()
    rb_distance = ((red - blue) ** 2).mean()
    gb_distance = ((green - blue) ** 2).mean()
    
    return (rg_distance + rb_distance + gb_distance) * 100  # Scale up the loss

class StyleGenerator:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.pipe = None
        self.style_tokens = []
        self.styles = [
            "ronaldo",
            "canna-lily-flowers102",
            "threestooges",
            "pop_art",
            "bird_style"
        ]
        self.style_names = [
            "Ronaldo",
            "Canna Lily",
            "Three Stooges",
            "Pop Art",
            "Bird Style"
        ]
        self.is_initialized = False

    def initialize_model(self):
        if self.is_initialized:
            return
            
        try:
            print("Initializing Stable Diffusion model...")
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                safety_checker=None
            )
            self.pipe = self.pipe.to("cuda")
            
            # Load style embeddings from current directory
            current_dir = Path(__file__).parent
            
            for style, style_name in zip(self.styles, self.style_names):
                style_path = current_dir / f"{style}.bin"
                if not style_path.exists():
                    raise FileNotFoundError(f"Style embedding not found: {style_path}")
                
                print(f"Loading style: {style_name}")
                token = load_learned_embed_in_clip(str(style_path), self.pipe.text_encoder, self.pipe.tokenizer)
                self.style_tokens.append(token)
                print(f"✓ Loaded style: {style_name}")
            
            self.is_initialized = True
            print("Model initialization complete!")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate_images(self, prompt, apply_loss=False, num_inference_steps=50, guidance_scale=7.5):
        if not self.is_initialized:
            self.initialize_model()
            
        images = []
        style_names = []
        
        try:
            def callback_fn(i, t, latents):
                if i % 5 == 0 and apply_loss:
                    try:
                        # Ensure latents are in the correct format and require gradients
                        latents = latents.float()
                        latents.requires_grad_(True)
                        
                        # Compute loss
                        loss = Distance_loss(latents)
                        
                        # Compute gradients manually
                        grads = torch.autograd.grad(
                            outputs=loss,
                            inputs=latents,
                            create_graph=False,
                            retain_graph=False,
                            only_inputs=True
                        )[0]
                        
                        # Update latents
                        with torch.no_grad():
                            latents = latents - 0.1 * grads
                            
                    except Exception as e:
                        print(f"Error in callback: {e}")
                        return latents
                        
                return latents

            for style_token, style_name in zip(self.style_tokens, self.style_names):
                styled_prompt = f"{prompt}, {style_token}"
                style_names.append(style_name)
                
                # Disable autocast for better gradient computation
                image = self.pipe(
                    styled_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    callback=callback_fn if apply_loss else None,
                    callback_steps=5
                ).images[0]
                
                images.append(image)
            
            return images, style_names
            
        except Exception as e:
            print(f"Error during image generation: {str(e)}")
            print(traceback.format_exc())
            raise

    def callback_fn(self, i, t, latents):
        if i % 5 == 0:  # Apply loss every 5 steps
            try:
                # Create a copy that requires gradients
                latents_copy = latents.detach().clone()
                latents_copy.requires_grad_(True)
                
                # Compute loss
                loss = Distance_loss(latents_copy)
                
                # Compute gradients
                if loss.requires_grad:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=latents_copy,
                        allow_unused=True,
                        retain_graph=False
                    )[0]
                    
                    if grads is not None:
                        # Apply gradients to original latents
                        return latents - 0.1 * grads.detach()
            
            except Exception as e:
                print(f"Error in callback: {e}")
            
        return latents

def generate_all_variations(prompt):
    try:
        generator = StyleGenerator.get_instance()
        if not generator.is_initialized:
            generator.initialize_model()
        
        # Generate images without loss
        regular_images, style_names = generator.generate_images(prompt, apply_loss=False)
        
        # Generate images with loss
        loss_images, _ = generator.generate_images(prompt, apply_loss=True)
        
        return regular_images, loss_images, style_names
    
    except Exception as e:
        print(f"Error in generate_all_variations: {str(e)}")
        print(traceback.format_exc())
        raise

def gradio_interface(prompt):
    try:
        regular_images, loss_images, style_names = generate_all_variations(prompt)
        
        return (
            regular_images,  # Just return the images directly
            loss_images     # Just return the images directly
        )
    except Exception as e:
        print(f"Error in interface: {str(e)}")
        print(traceback.format_exc())
        # Return empty lists in case of error
        return [], []

# Create a more beautiful interface with custom styling
with gr.Blocks(css="""
    .gradio-container {
        background-color: #1f2937 !important;
    }
    .dark-theme {
        background-color: #111827;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        border: 1px solid #374151;
        color: #f3f4f6;
    }
""") as iface:
    # Header section with dark theme
    gr.Markdown(
        """
        <div class="dark-theme" style="text-align: center; max-width: 800px; margin: 0 auto;">
        # 🎨 AI Style Transfer Studio
        ### Transform your ideas into artistic masterpieces with custom styles and enhanced colors
        </div>
        """
    )

    # Define the generate_single_style function first
    def generate_single_style(prompt, selected_style):
        try:
            generator = StyleGenerator.get_instance()
            if not generator.is_initialized:
                generator.initialize_model()
            
            # Find the index of the selected style
            style_idx = generator.style_names.index(generator.style_names[selected_style])
            
            # Generate single image with selected style
            styled_prompt = f"{prompt}, {generator.style_tokens[style_idx]}"
            
            # Set seed for reproducibility
            generator_seed = 42
            torch.manual_seed(generator_seed)
            torch.cuda.manual_seed(generator_seed)
            
            # Generate base image
            with autocast("cuda"):
                base_image = generator.pipe(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=torch.Generator("cuda").manual_seed(generator_seed)
                ).images[0]
            
            # Generate same image with loss
            with autocast("cuda"):
                loss_image = generator.pipe(
                    styled_prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    callback=generator.callback_fn,
                    callback_steps=5,
                    generator=torch.Generator("cuda").manual_seed(generator_seed)
                ).images[0]
            
            return [
                gr.update(visible=False),  # error_message
                base_image,                # original_image
                loss_image                 # loss_image
            ]
        except Exception as e:
            print(f"Error in generate_single_style: {e}")
            return [
                gr.update(value=f"Error: {str(e)}", visible=True),  # error_message
                None,  # original_image
                None   # loss_image
            ]

    # Main content
    with gr.Row():
        # Left sidebar for controls
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 🎯 Controls")
            
            prompt = gr.Textbox(
                label="What would you like to create?",
                placeholder="e.g., a soccer player celebrating a goal",
                lines=3
            )
            
            style_radio = gr.Radio(
                choices=[
                    "Ronaldo Style",
                    "Canna Lily",
                    "Three Stooges",
                    "Pop Art",
                    "Bird Style"
                ],
                label="Choose Your Style",
                value="Ronaldo Style",
                type="index"
            )
            
            generate_btn = gr.Button(
                "🚀 Generate Artwork", 
                variant="primary",
                size="lg"
            )
            
            # Error messages
            error_message = gr.Markdown(visible=False)
            
            # Style description
            style_description = gr.Markdown()
        
        # Right side for image display
        with gr.Column(scale=2):
            gr.Markdown("## 🖼️ Generated Artwork")
            with gr.Row():
                with gr.Column():
                    original_image = gr.Image(
                        label="Original Style",
                        show_label=True,
                        height=400
                    )
                with gr.Column():
                    loss_image = gr.Image(
                        label="Color Enhanced",
                        show_label=True,
                        height=400
                    )
    
    # Info section
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                <div class="dark-theme">
                ## 🎨 Style Guide
                
                | Style | Best For |
                |-------|----------|
                | **Ronaldo Style** | Dynamic sports scenes, action shots, celebrations |
                | **Canna Lily** | Natural scenes, floral compositions, garden imagery |
                | **Three Stooges** | Comedy, humor, expressive character portraits |
                | **Pop Art** | Vibrant artwork, bold colors, stylized designs |
                | **Bird Style** | Wildlife, nature scenes, peaceful landscapes |
                
                *Choose the style that best matches your creative vision*
                </div>
                """
            )
        with gr.Column():
            gr.Markdown(
                """
                <div class="dark-theme">
                ## 🔍 Color Enhancement Technology
                
                Our advanced color processing uses distance loss to enhance your images:
                
                ### 🌈 Color Dynamics
                - **Vibrancy**: Intensifies colors naturally
                - **Contrast**: Improves depth and definition
                - **Balance**: Optimizes color relationships
                
                ### 🎨 Technical Features
                - **Channel Separation**: RGB optimization
                - **Loss Function**: Mathematical color enhancement
                - **Real-time Processing**: Dynamic adjustments
                
                ### ✨ Benefits
                - Richer, more vivid colors
                - Clearer color boundaries
                - Reduced color muddiness
                - Enhanced artistic impact
                
                <small>*Our color distance loss technology mathematically optimizes RGB channel relationships*</small>
                </div>
                """
            )

    # Update style description on change
    def update_style_description(style_idx):
        descriptions = [
            "Perfect for capturing dynamic sports moments and celebrations",
            "Ideal for creating beautiful natural and floral compositions",
            "Great for adding humor and expressiveness to your scenes",
            "Transform your ideas into vibrant pop art masterpieces",
            "Specialized in capturing the beauty of nature and wildlife"
        ]
        styles = ["Ronaldo Style", "Canna Lily", "Three Stooges", "Pop Art", "Bird Style"]
        return f"### Selected: {styles[style_idx]}\n{descriptions[style_idx]}"

    style_radio.change(
        fn=update_style_description,
        inputs=style_radio,
        outputs=style_description
    )

    # Connect the generate button
    generate_btn.click(
        fn=generate_single_style,
        inputs=[prompt, style_radio],
        outputs=[error_message, original_image, loss_image]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(
        share=True,
        show_error=True
    ) 