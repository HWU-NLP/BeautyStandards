# from diffusers import AutoPipelineForText2Image

# def setup_realistic_vision_model(model_path=None):
#     # Set up the Realistic Vision V5.1 model
#     print("\n#### Loading Realistic Vision model")
    
#     model_id = model_path or "SG161222/Realistic_Vision_V5.1_noVAE"
    
#     # Load the pipeline
#     pipe = AutoPipelineForText2Image.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16 if USE_HALF_PRECISION and DEVICE == "cuda" else torch.float32,
#         safety_checker=None,
#         use_safetensors=True,
#     )
    
#     return pipe

from pathlib import Path
import argparse
import os
import torch
import time
from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from transformers import T5EncoderModel
from huggingface_hub import login
from tqdm import tqdm
import json

def setup_sd35_model(model_path=None):
    print("Loading Stable Diffusion 3.5 Large model")
    model_id = model_path or "stabilityai/stable-diffusion-3.5-large"
    
    # Configure 4-bit quantization (NF4)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load quantized transformer model
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        use_auth_token=True
    )
    
    # Load quantized T5 encoder
    t5_nf4 = T5EncoderModel.from_pretrained(
        "diffusers/t5-nf4", 
        torch_dtype=torch.bfloat16
    )
    
    # Create pipeline with quantized components
    # This version of pipe still defaults to CLIP, which truncates the prompt at 70 tokens
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        use_auth_token=True
    )

    return pipe

def generate_realistic_images(prompt_file, output_dir="generated_images", model_path=None):
    """Generate photorealistic images using SD 3.5 (portrait aspect ratio)"""
    # Parameters optimized for photorealistic portraits based on Replicate example
    # Blog post: https://replicate.com/blog/get-the-best-from-stable-diffusion-3
    # https://replicate.com/p/pg4r85kd39rgm0cg5dp9e8q56c?input=json
    width = 832
    height = 1216
    steps = 28
    guidance_scale = 3.5
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = setup_sd35_model(model_path)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(DEVICE)
    
    # Load prompts based on file type
    with open(prompt_file, "r", encoding="utf-8") as file:
        prompts = []
        for line in file:
            parts = line.strip().split("|")
            if "output_text_selfBleu" in prompt_file and len(parts) >= 4:
                prompts.append(parts[3].strip())  # LLM output
            elif "image_prompts" in prompt_file and len(parts) >= 2:
                prompts.append(parts[1].strip())  # prompt
            else:
                continue #skip blank lines etc
    
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")
    print(f"Generating images with: {steps} steps, {guidance_scale} guidance, {width}x{height} resolution")
    
    # Create metadata file to track generations
    metadata_file = output_dir / "generation_metadata.jsonl"
    
    with open(metadata_file, "w", encoding="utf-8") as outfile:
        # Process each prompt
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            actual_idx = idx
            # Add photo directive
            # Following blog recommendation to skip negative prompt
            # enhanced_prompt = f"A professional photograph of {prompt}, detailed, high resolution"
            # enhanced_prompt = f"A professional photograph of {prompt}, natural composition, uncropped"
            enhanced_prompt = f"A full length realistic photo of {prompt}"

            # Generate the image
            generation_output = pipe(
                prompt="",#give empty to satisfy pipeline requirement of having prompt for CLIP
                prompt_2=enhanced_prompt, #prompt defaults to CLIP encoder that truncates the input, prompt_embeds_2 directly embeds using the  T5-based encoder
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_type="pil",
                max_sequence_length=512  # Enable long prompts processing
            )
            
            # Save the image
            image = generation_output.images[0]
            timestamp = int(time.time())
            source_type = "lm" if "output_text_selfBleu" in prompt_file else "base"
            filename = f"{source_type}_sd3-5_{actual_idx}.png"
            # filename = f"person_{idx+1}_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            image.save(save_path)
            
            print(f"Generated: {filename}")
            
            # Save metadata
            metadata = {
                "model": "SD3.5Large",
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "filename": filename,
                "path": str(save_path),
                "parameters": {
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height
                }
            }
            outfile.write(json.dumps(metadata) + "\n")
            outfile.flush()
    
    print(f"Generation complete! {len(prompts)} images generated.")
    print(f"Output saved to {output_dir}")
    print(f"Metadata saved to {metadata_file}")

def main():
    # Authenticate with Hugging Face
    hub_token = ""
    if hub_token:
        login(token=hub_token)
    else:
        print("Warning: No Hugging Face token provided. Using public access.")

    parser = argparse.ArgumentParser(description="Generate high-quality images with Stable Diffusion 3.5")
    #parser.add_argument("prompt_file", type=str, help="Path to file containing prompts")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to SD 3.5 model")
    args = parser.parse_args()
    
    generate_realistic_images("output_text_selfBleu.txt", args.output_dir, args.model_path)

if __name__ == "__main__":
    main()

