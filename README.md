# Stable Diffusion Style Transfer with Color Distance Loss

This project implements a Stable Diffusion-based image generation system with custom style transfers and color enhancement through a distance loss function.

## 🎨 Features

### Style Transfer
- Implements 5 different artistic styles:
  - Ronaldo Style (Sports scenes)
  - Canna Lily (Nature and flowers)
  - Three Stooges (Comedy scenes)
  - Pop Art (Vibrant artistic style)
  - Bird Style (Wildlife imagery)

### Color Distance Loss
The project implements a unique color enhancement through RGB channel separation:
- Calculates distances between RGB channels
- Enhances color vibrancy and contrast
- Creates more distinct color separation
- Reduces color mixing and muddiness

### Interactive Interface
- User-friendly Gradio web interface
- Side-by-side comparison of original and enhanced images
- Real-time style selection
- Example prompts for each style

## 🚀 Setup and Installation

1. Install dependencies:
bash
pip install -r requirements.txt

2. Required files:
- Style embeddings (.bin files):
  - ronaldo.bin
  - canna-lily-flowers102.bin
  - threestooges.bin
  - pop_art.bin
  - bird_style.bin

3. Run the application:
bash
python app.py

## 🎮 Usage

1. Enter a text prompt describing your desired image
2. Select a style using the radio buttons
3. Click "Generate Images" to create:
   - Left: Original styled image
   - Right: Image with color distance loss applied

## 🔧 Technical Details

### Distance Loss Function
def Distance_loss(images):
# Extract RGB channels
red = images[:,0:1]
green = images[:,1:2]
blue = images[:,2:3]
# Calculate channel distances
rg_distance = ((red - green) 2).mean()
rb_distance = ((red - blue) 2).mean()
gb_distance = ((green - blue) 2).mean()
return (rg_distance + rb_distance + gb_distance) 100

This loss function:
- Separates RGB channels
- Calculates squared distances between channels
- Enhances color distinctiveness
- Applies during the generation process

## 📝 Example Prompts

- Sports: "a soccer player celebrating a goal"
- Nature: "beautiful flowers in a garden"
- Comedy: "three comedians performing a skit"
- Art: "a colorful portrait in pop art style"
- Wildlife: "birds flying in a natural landscape"

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Transformers
- Gradio
- CUDA-capable GPU recommended

## 📊 Results

The color distance loss typically produces:
- More vibrant colors
- Better color separation
- Enhanced contrast
- More distinctive style characteristics

## 🤝 Contributing

Feel free to:
- Submit issues
- Fork the repository
- Submit pull requests

## 📜 License

This project is open-source and available under the MIT License.