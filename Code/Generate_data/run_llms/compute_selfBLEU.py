import argparse
import csv
import os
import re
from datasets import load_metric
# import evaluate 
from tqdm import tqdm

def extract_llm_name(filename):
    """
    Extract LLM name from the filename
    :param filename: Path to the file
    :return: LLM name
    """
    # Get base filename without path and extension
    base_name = os.path.basename(filename)
    
    # Remove extension
    name_without_ext = os.path.splitext(base_name)[0]
    
    # If the filename contains 'output_text_' prefix, extract what follows
    if 'output_text_' in name_without_ext:
        return name_without_ext.split('output_text_')[1]
    
    return name_without_ext

def count_tokens(text):
    """
    Count the number of tokens in the text (simple whitespace-based tokenization)
    :param text: Input text
    :return: Number of tokens
    """
    if not text:
        return 0
    return len(text.split())

def parse_output_file(file_path):
    """
    Parse the output file using CSV reader and return a dictionary with idx as key
    :param file_path: Path to the output file
    :return: Dictionary with idx as key and (prompt, selected, output) as value
    """
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        # Use csv reader with pipe delimiter and strip whitespace
        reader = csv.reader((line.strip() for line in f), delimiter='|')
        for row in reader:
            if len(row) >= 4:  # Ensure we have enough columns
                idx = row[0].strip()
                prompt = row[1].strip()
                selected = row[2].strip()
                output = row[3].strip()
                data[idx] = (prompt, selected, output)
    
    return data

def select_lowest_self_bleu(output_files, llm_names):
    """
    Select the output with the lowest self-BLEU score for each prompt
    :param output_files: List of paths to output files
    :param llm_names: List of LLM names
    :return: Dictionary with selected outputs
    """
    # Load BLEU metric from HuggingFace
    bleu = load_metric("bleu")

    # # Load the BLEU metric from the evaluate library
    # bleu = evaluate.load("bleu")
    
    # Parse all output files
    all_data = []
    for file_path in output_files:
        all_data.append(parse_output_file(file_path))
    
    # Create a dictionary to store outputs by idx and LLM
    outputs_by_idx = {}
    for idx_llm, data in enumerate(all_data):
        for idx, (prompt, selected, output) in data.items():
            if idx not in outputs_by_idx:
                outputs_by_idx[idx] = []
            outputs_by_idx[idx].append((idx_llm, output))
    
    # Calculate self-BLEU scores and select the lowest for each idx
    selected_outputs = {}
    
    for idx, outputs in tqdm(outputs_by_idx.items(), desc="Calculating self-BLEU scores"):
        # Check if any output is very short
        if any(count_tokens(output) < 20 for _, output in outputs):
            # Just pick the longest output in this case
            best_idx = max(range(len(outputs)), key=lambda i: len(outputs[i][1])) # just take the longest string.
            llm_idx = outputs[best_idx][0]
            selected_outputs[idx] = (all_data[0].get(idx)[0], all_data[0].get(idx)[1], 
                                    outputs[best_idx][1], llm_idx, llm_names[llm_idx])
            continue
        
        best_llm_idx = None
        lowest_self_bleu = float('inf')
        
        for i, (llm_idx, output) in enumerate(outputs):
            # Get the outputs from other LLMs
            references = [out for j, (_, out) in enumerate(outputs) if j != i]
            
            # Tokenize
            candidate = output.split()
            refs_tokenized = [ref.split() for ref in references]
            
            # Calculate BLEU score
            try:
                score = bleu.compute(predictions=[candidate], references=[refs_tokenized])['bleu']
                if score < lowest_self_bleu:
                    lowest_self_bleu = score
                    best_llm_idx = llm_idx
            except:
                # In case of errors (e.g., empty outputs), set score to 0
                if lowest_self_bleu == float('inf'):
                    best_llm_idx = llm_idx
        
        # Get the prompt and selected from any of the data dictionaries
        prompt = all_data[0].get(idx)[0]
        selected = all_data[0].get(idx)[1]
        
        # Store the selected output
        # for llm_idx, output in outputs:
        #     if llm_idx == best_llm_idx:
        #         best_output = output
        #         break  # Stop as soon as we find the matching model
        best_output = next((output for llm_idx, output in outputs if llm_idx == best_llm_idx), "")
        selected_outputs[idx] = (prompt, selected, best_output, best_llm_idx, llm_names[best_llm_idx])
    
    return selected_outputs

def write_output_file(selected_outputs, output_path):
    """
    Write the selected outputs to a file using CSV writer
    :param selected_outputs: Dictionary with selected outputs
    :param output_path: Path to the output file
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for idx in sorted(selected_outputs.keys(), key=int):
            prompt, selected, output, _, llm_name = selected_outputs[idx]
            writer.writerow([idx, prompt, selected, output, llm_name])

def main():
    parser = argparse.ArgumentParser(description='Select outputs with lowest self-BLEU scores')
    parser.add_argument('--input-files', nargs='+', required=True, help='Paths to input files')
    parser.add_argument('--output-file', required=True, help='Path to output file')
    
    args = parser.parse_args()
    
    # Extract LLM names from filenames
    llm_names = [extract_llm_name(file_path) for file_path in args.input_files]
    print(f"Detected LLM names: {llm_names}")
    
    selected_outputs = select_lowest_self_bleu(args.input_files, llm_names)
    write_output_file(selected_outputs, args.output_file)
    
    print(f"Selected outputs written to {args.output_file}")

if __name__ == "__main__":
    main()