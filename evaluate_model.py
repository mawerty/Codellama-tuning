import os
import torch
import json
import time
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import evaluate

# --- Config ---
BASE_MODEL_ID = "codellama/CodeLlama-7b-Python-hf"
ADAPTER_PATH_7B = "./requests_codellama_final"
TEST_DATASET_PATH = "data/test.jsonl"
RESULTS_DIR = "results"

use_8bit_quantization = True
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.6
TOP_P = 0.9
DO_SAMPLE = True

def load_tokenizer(model_id):
    """Loads the tokenizer."""
    print(f"Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model_and_tokenizer(model_choice):
    """Loads the specified model and its tokenizer."""
    adapter_path = None
    is_finetuned = False
    if model_choice == "finetuned_7b":
        base_model_id = "codellama/CodeLlama-7b-Python-hf"; adapter_path = ADAPTER_PATH_7B; is_finetuned = True
    elif model_choice == "base_7b": base_model_id = "codellama/CodeLlama-7b-Python-hf"
    elif model_choice == "base_13b": base_model_id = "codellama/CodeLlama-13b-Python-hf"
    else: raise ValueError(f"Invalid model choice: {model_choice}")
    print(f"\nLoading {model_choice} (Base ID: {base_model_id})...")
    start_load_time = time.time()
    tokenizer = load_tokenizer(base_model_id)
    bnb_config = BitsAndBytesConfig(load_in_8bit=use_8bit_quantization) if use_8bit_quantization else None
    dtype = compute_dtype
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=dtype)
    if is_finetuned:
        print(f"Applying LoRA adapter from: {adapter_path}")
        adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
        adapter_model_safetensors = os.path.join(adapter_path, "adapter_model.safetensors")
        adapter_model_bin = os.path.join(adapter_path, "adapter_model.bin")
        if not (os.path.exists(adapter_model_safetensors) or os.path.exists(adapter_model_bin)) or not os.path.exists(adapter_config_file):
             raise FileNotFoundError(f"Adapter files (.safetensors or .bin, and .json) not found in '{adapter_path}'. Check path.")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("LoRA adapter applied.")
    else:
        model = base_model
        print("Using base model directly.")
    model.eval()
    end_load_time = time.time()
    print(f"Model loaded in {end_load_time - start_load_time:.2f} seconds.")
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, is_finetuned):
    """Generates a completion for a given prompt and returns completion + duration."""
    start_gen_time = time.time() # Start timing *before* generation
    device = model.device
    if is_finetuned: input_text = f"Prompt: {prompt}\nCompletion:"
    else: input_text = f"Instruction:\n{prompt}\n\nPython Code:\n```python\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            top_p=TOP_P, do_sample=DO_SAMPLE, pad_token_id=tokenizer.eos_token_id
        )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if "```" in completion: completion = completion.split("```")[0]
    completion = completion.strip()
    end_gen_time = time.time() # End timing *after* generation
    duration = end_gen_time - start_gen_time
    return completion, duration # Return duration

# --- Main Execution ---

def main(args):
    start_total_time = time.time()
    model_choice = args.model

    # --- Setup ---
    print(f"Starting Evaluation for '{model_choice}'...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_filename = os.path.join(RESULTS_DIR, f"evaluation_outputs_{model_choice}.jsonl")
    metrics_filename = os.path.join(RESULTS_DIR, f"evaluation_metrics_{model_choice}.json")

    # --- Load Data ---
    print(f"\nLoading test dataset from: {TEST_DATASET_PATH}")
    if not os.path.exists(TEST_DATASET_PATH): raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET_PATH}")
    test_dataset = load_dataset('json', data_files=TEST_DATASET_PATH, split='train')
    total_examples = len(test_dataset)
    print(f"Loaded {total_examples} examples for testing.")

    # --- Load Model & Tokenizer ---
    model, tokenizer = load_model_and_tokenizer(model_choice)
    is_finetuned_model = (model_choice == "finetuned_7b")

    # --- Generate Outputs ---
    results = []
    generation_times = [] # Store individual generation times
    print(f"\n--- Generating with {model_choice} ---")
    start_gen_loop_time = time.time()
    try:
        for i, example in enumerate(test_dataset):
            prompt = example['prompt']
            reference = example['completion']

            # --- Print BEFORE starting generation ---
            print(f"Processing example {i+1}/{total_examples}...")

            generated_output, duration = generate_completion(model, tokenizer, prompt, is_finetuned=is_finetuned_model)
            generation_times.append(duration) # Store duration

            # --- Print AFTER finishing generation ---
            print(f"  Example {i+1}/{total_examples} finished in {duration:.2f} seconds.")

            results.append({
                "index": i, "prompt": prompt, "reference": reference, "output": generated_output
            })
        print(f"\nGeneration loop finished.")
        end_gen_loop_time = time.time()
        total_generation_time = end_gen_loop_time - start_gen_loop_time
        avg_time_per_example = total_generation_time / total_examples if total_examples > 0 else 0
        print(f"Total generation time: {total_generation_time:.2f} seconds ({avg_time_per_example:.2f} s/example avg).")

        # --- Save Results ---
        print(f"\nSaving raw outputs to: {output_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results: f.write(json.dumps(result) + '\n')
        print("Outputs saved successfully.")

        # --- Calculate and Save Metrics ---
        print("\nCalculating metrics (BLEU, CER, ROUGE, ChrF)...")
        try:
            # Load metric calculators here (inside try block)
            bleu_metric = evaluate.load("bleu")
            cer_metric = evaluate.load("cer")
            rouge_metric = evaluate.load("rouge")
            chrf_metric = evaluate.load("chrf") # Default is chrf, sometimes called chrf++

            predictions = [res["output"] for res in results]
            references_list_for_bleu = [[res["reference"]] for res in results]
            references_str = [res["reference"] for res in results]

            bleu_score = bleu_metric.compute(predictions=predictions, references=references_list_for_bleu)
            cer_score = cer_metric.compute(predictions=predictions, references=references_str)
            rouge_scores = rouge_metric.compute(predictions=predictions, references=references_str)
            # Standard ChrF (score) or ChrF++ (score) - evaluate library default is usually good
            chrf_score = chrf_metric.compute(predictions=predictions, references=references_str)

            print(f"  BLEU Score: {bleu_score['bleu']:.4f}")
            print(f"  CER Score: {cer_score:.4f}") # Lower is better
            print(f"  ROUGE-L Score: {rouge_scores['rougeL']:.4f}")
            print(f"  ChrF Score: {chrf_score['score']:.4f}")

            metrics = {
                "model_choice": model_choice,
                "num_examples": total_examples,
                "bleu": bleu_score['bleu'],
                "cer": cer_score,
                "rougeL": rouge_scores['rougeL'],
                "chrf": chrf_score['score'],
                "generation_time_s": total_generation_time,
                "avg_generation_time_s": avg_time_per_example
            }

            print(f"\nSaving metrics to: {metrics_filename}")
            with open(metrics_filename, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
            print("Metrics saved.")

        except AttributeError as e:
             print(f"\nError calculating metrics: {e}")
             print("This likely means the 'evaluate' library is not installed correctly or there's a naming conflict.")
             print("Please ensure your script is NOT named 'evaluate.py' and try reinstalling:")
             print("`pip uninstall evaluate -y && pip install evaluate sacrebleu jiwer rouge-score`")
        except Exception as e:
            print(f"\nError calculating or saving metrics: {e}")
            print("Skipping metrics. Ensure 'evaluate', 'sacrebleu', 'jiwer', 'rouge-score' are installed.")


    except Exception as e:
        print(f"\nAn error occurred during the evaluation process: {e}")
    finally:
        # --- Unload Model and Clear Cache ---
        print("\nUnloading model and clearing cache...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        print("Model unloaded.")

    end_total_time = time.time()
    print(f"\nEvaluation script for '{model_choice}' finished in {end_total_time - start_total_time:.2f} seconds.")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base or fine-tuned CodeLlama models.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['base_7b', 'base_13b', 'base_34b', 'finetuned_7b'],
        help="Specify model version: 'base_7b', 'base_13b', 'base_34b', or 'finetuned_7b'."
    )
    parsed_args = parser.parse_args()
    main(parsed_args)