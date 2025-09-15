import argparse
from pathlib import Path
import pandas as pd
import torch
from f_lite import FLitePipeline

# Trick required because it is not a native diffusers model
from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES, ALL_IMPORTABLE_CLASSES

LOADABLE_CLASSES["f_lite"] = LOADABLE_CLASSES["f_lite.model"] = {"DiT": ["save_pretrained", "from_pretrained"]}
ALL_IMPORTABLE_CLASSES["DiT"] = ["save_pretrained", "from_pretrained"]


def generate_realistic_images(prompt_file, prompt_type, output_dir, model_path):
    print(f"Loading {model_path} model")
    
    # Load the prompts
    prompts = pd.read_csv(prompt_file, sep="|", header=None)

    # Load the pipeline
    pipeline = FLitePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # pipeline.enable_model_cpu_offload() # For less memory consumption. Alternatively, pipeline.to("cuda")
    pipeline.to("cuda")
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    for i in range(prompts.shape[0]):
        index = prompts.iloc[i, 0]
        if prompt_type == "lm":
            prompt = prompts.iloc[i, 3]  # LLM output
        elif prompt_type == "base":
            prompt = prompts.iloc[i, 1]
        else:
            raise ValueError("Invalid prompt type. Use 'base' or 'lm'.")
        print(f"Generating image {index}/{prompts.shape[0]}: {prompt}")

        enhanced_prompt = f"A professional photograph of {prompt}, natural composition, uncropped"
        # enhanced_prompt = f"A photograph of {prompt}, natural composition, uncropped" # -> discard this version
        # enhanced_prompt = f"A full length realistic photo of: {prompt}"  # -> final prompt version to be used
        
        # Generate an image
        output = pipeline(
            prompt=enhanced_prompt,
            height=1344,
            width=896,
            num_inference_steps=28,
            guidance_scale=3.5,  # Guidance scale for the diffusion process
            negative_prompt=None,
        )

        # Save the generated image
        output.images[0].save(Path(output_dir, f"{prompt_type}_fp_{index}.png"))
        

def main():
    # # Authenticate with Hugging Face
    # hub_token = ""
    # if hub_token:
    #     login(token=hub_token)
    # else:
    #     print("Warning: No Hugging Face token provided. Using public access.")

    parser = argparse.ArgumentParser(description="Generate high-quality images with Stable Diffusion 3.5")
    parser.add_argument("--prompt_file", type=str, help="Path to file containing prompts")
    parser.add_argument("--prompt_type", type=str, help="Prompt type: base, or lm")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to SD 3.5 model")
    args = parser.parse_args()
    
    generate_realistic_images(args.prompt_file, args.prompt_type, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()

"""
Sample usage

# run freepik with base prompts
python run_freepik.py --prompt_file=/users/aj2066/sharedscratch/BeautyStandards/Code/Generate_data/run_image_models/image_prompts.txt \
        --prompt_type=base \
        --output_dir=/users/aj2066/sharedscratch/BeautyStandards/Code/Generate_data/run_image_models/images \
        --model_path=Freepik/F-Lite

# run freepik with llm-generated descriptions
python run_freepik.py --prompt_file=/users/aj2066/sharedscratch/BeautyStandards/Code/Generate_data/run_image_models/output_text_selfBleu.txt \
        --prompt_type=lm \
        --output_dir=/users/aj2066/sharedscratch/BeautyStandards/Code/Generate_data/run_image_models/images \
        --model_path=Freepik/F-Lite
"""