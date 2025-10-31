# BeautyStandards
Sources for "Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models"

## 📁 Repository Structure

```
BeautyStandards/
│
├── Prompt_template/
│   ├── beauty_taxonomy.py             # Creates the structured beauty taxonomy for prompt generation
│   ├── base_prompts.txt               # Core set of base prompt templates
│   ├── prompts_metadata.json          # Metadata associated with taxonomy prompts
│   ├── llama3.1_prompts.txt           # Prompts for LLaMA 3.1 model
│   ├── llama3.1_instruct_prompts.txt  # Prompts for LLaMA 3.1 Instruct model
│   ├── deepseek_llm_prompts.txt       # Prompts for DeepSeek LLM
│   └── image_prompts.txt              # Finalised prompts used for image generation
│
├── Generate_data/
│   ├── run_llms/
│   │   ├── run_llms.py                # Runs LLMs with taxonomy-based prompts
│   │   ├── compute_selfBLEU.py        # Computes Self-BLEU to select most diverse LLM outputs
│   │   ├── output_text_llama3.1.txt   # LLaMA 3.1 generated text outputs
│   │   ├── output_text_llama3.1_instruct.txt
│   │   ├── output_text_deepseek_llm.txt # DeepSeek LLM outputs
│   │   └── output_text_selfBleu.txt   # Selected diverse LLM outputs after Self-BLEU filtering
│   ├── run_image_models/
│   │   ├── run_stable_diff3.5_full.py # Generates images via Stable Diffusion 3.5
│   │   ├── run_freepik.py             # Generates images via Freepik API
│   │   └── image_prompts.txt          # Input prompts for image generation
│
├── Results/
│   ├── analyse.ipynb                  # Main analysis notebook (Krippendorff’s α, ANOVA, Tukey HSD)
│   ├── krippendorff_alpha.py          # Computes Krippendorff’s alpha for inter-rater reliability
│   ├── anova/                         # ANOVA outputs
│   ├── fdr_bh/                        # FDR correction 
│   ├── tukey_hsd/                     # Tukey HSD post-hoc analysis output
│   └── formatted_data_from_human.tsv  # Annotated dataset with anonymised Prolific IDs
│
├── LICENSE
└── README.md

```
