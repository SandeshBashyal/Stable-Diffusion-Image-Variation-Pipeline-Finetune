from glob import glob
import os
test_file = '/home/rmuproject/rmuproject/users/sandesh/PatternVerse Test Set - I/*'
test_list = glob(test_file, recursive = True)
test_list = sorted(test_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
from diffusers import StableDiffusionImageVariationPipeline
import torch
from PIL import Image
# Load the pre-trained model
model_id = "/home/rmuproject/rmuproject/users/sandesh/models/80_epochs"
pipeline = StableDiffusionImageVariationPipeline.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
pipeline.enable_model_cpu_offload()
output_path = "/home/rmuproject/rmuproject/users/sandesh/renders"
seed = 42
num_variations = 5
generator = torch.Generator(device="cuda").manual_seed(seed)
image_folder = os.path.join(output_path, "Output")
input_image_folder = os.path.join(output_path, "Input")
alphabets = ['a', 'b', 'c', 'd', 'e']
os.makedirs(image_folder, exist_ok=True)
os.makedirs(input_image_folder, exist_ok=True)
for i, input_path in enumerate(test_list):
    input_image = Image.open(input_path).convert("RGB")
    input_image.save(f'{input_image_folder}/input_{i + 1}.png')
    width, height = input_image.size
    print(f"Original size: {width}x{height}")
    # Resize the image to a size the model supports (typically 512x512 or similar)
    input_image_resized = input_image.resize((512, 512))
    generated_images = []
    for _ in range(num_variations):
        images = pipeline(
            image=input_image_resized,
            num_inference_steps=50,  # Number of diffusion steps
            guidance_scale=2.5,  # Scale for conditional guidance
            generator=generator
        ).images
        generated_images.append(images[0])
    
    # Save the generated images with the new dimensions
    
    for j, img in enumerate(generated_images):
        # Resize the generated images to the output dimension
        resized_image = img.resize((width, height))
        resized_image.save(f"{image_folder}/output_{i+1}{alphabets[j]}.png")
