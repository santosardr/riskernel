import os
import sys
import json
import argparse
import time
import torch
from tqdm import tqdm
import csv

# Add riskernel to path to import inference_ris_v3 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from inference_ris_v3 import load_ris_model, ARTICLES_DIR
except ImportError as e:
    import traceback
    print("=== IMPORT ERROR TRACEBACK ===")
    traceback.print_exc()
    print("==============================")
    print("Could not import inference_ris_v3. Ensure this script is in riskernel/benchmark/")
    sys.exit(1)

def get_option_token_ids(tokenizer):
    """
    Get token IDs for options A, B, C, D, E.
    We check multiple valid representations just in case.
    """
    options = ["A", "B", "C", "D", "E"]
    token_dict = {}
    for opt in options:
        # Some tokenizers prefix with space depending on prompt. We evaluate strictly 'A'
        token_id = tokenizer.convert_tokens_to_ids(opt)
        if token_id is None or token_id == tokenizer.unk_token_id:
            # Revert to encoding
            encoded = tokenizer.encode(opt, add_special_tokens=False)
            token_id = encoded[0] if encoded else None
        token_dict[opt] = token_id
    return token_dict

def extract_qa_logits(model, tokenizer, prompt, past_key_values, option_token_ids):
    # 1. Capture the original length of the base context
    original_len = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            original_len = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
            original_len = past_key_values[0][0].shape[-2]

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values, 
            use_cache=True
        )
        
    # 2. STATUTORY ISOLATION: Crop the cache back to original length
    # This prevents Questions from leaking into each other and from exceeding RIS mask bounds.
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        new_pkv = outputs.past_key_values
        if hasattr(new_pkv, "crop"):
            new_pkv.crop(original_len)
        elif hasattr(new_pkv, "key_cache"): # Manual Truncation for DynamicCache
            for i in range(len(new_pkv.key_cache)):
                new_pkv.key_cache[i] = new_pkv.key_cache[i][:, :, :original_len, :]
                new_pkv.value_cache[i] = new_pkv.value_cache[i][:, :, :original_len, :]
        # If it's a legacy tuple, it didn't modify in-place anyway.
        
    next_token_logits = outputs.logits[0, -1, :]
    
    probs = {}
    for letter, tok_id in option_token_ids.items():
        if tok_id:
            probs[letter] = next_token_logits[tok_id].item()
        else:
            probs[letter] = float('-inf')
            
    # Softmax specific options to normalize
    prob_tensor = torch.tensor(list(probs.values()))
    softmax_probs = torch.nn.functional.softmax(prob_tensor, dim=0).tolist()
    
    normalized_probs = {list(probs.keys())[i]: softmax_probs[i] for i in range(5)}
    predicted_option = max(normalized_probs, key=normalized_probs.get)
    
    return predicted_option, normalized_probs

def chunked_prefill(model, tokenizer, context_text, max_len, chunk_size=1024):
    """Prefills the context safely into the KV cache."""
    inputs = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = inputs['input_ids']
    total_tokens = input_ids.shape[1]
    
    # Strictly enforce window sizing (leave 1024 for chat)
    alloc_tokens = min(total_tokens, max_len - 1024)
    if alloc_tokens <= 0:
        alloc_tokens = 0
        
    truncated_input_ids = input_ids[:, :alloc_tokens]
    
    past_key_values = None
    print(f"[PREFILL] Ingesting {alloc_tokens} context tokens... (Max Window: {max_len})")
    
    if alloc_tokens == 0:
        return None, 0
        
    with torch.no_grad():
        for i in tqdm(range(0, alloc_tokens, chunk_size), desc="Prefill"):
            end_idx = min(i + chunk_size, alloc_tokens)
            chunk = truncated_input_ids[:, i:end_idx]
            outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
    return past_key_values, alloc_tokens

def main():
    parser = argparse.ArgumentParser(description="RIS-Kernel Automated Benchmark (Discriminative QA)")
    parser.add_argument("--windows", type=str, default="0,4096,8192,16384,32768,65536", help="Comma-separated window sizes. 0 = Baseline (No Context)")
    parser.add_argument("--densities", type=str, default="0.05", help="Comma-separated densities")
    parser.add_argument("--models", type=str, default="qwen2", help="Comma-separated models: qwen2,tinyllama")
    parser.add_argument("--n_seeds", type=str, default="1", help="Comma-separated seed counts (ensemble)")
    parser.add_argument("--qa_dataset", type=str, default="scripts/benchmark/qa_dataset.json", help="Path to JSON questions")
    parser.add_argument("--context_files", type=str, default="scripts/data/genppi.txt,scripts/data/aom.txt,scripts/data/ajinshanensis.txt,scripts/data/meta.txt", help="Comma-separated files")
    parser.add_argument("--out_csv", type=str, default="scripts/benchmark/results_64.csv", help="Output file")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick baseline+4k sweep with first few Qs")
    parser.add_argument("--rope_type", type=str, default="linear", help="Type of RoPE scaling (linear or yarn)")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use")
    parser.add_argument("--ris_mode", type=str, default="stochastic", choices=["stochastic", "structural"], help="RIS geometry mode")
    parser.add_argument("--b_max", type=int, default=2048, help="Max block size for structural RIS mode. Use int(seq_len * max_density) to control the density ceiling.")
    
    args = parser.parse_args()

    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)

    if args.threads:
        print(f"[SYSTEM] Capping execution to {args.threads} threads (handled by inference engine).")
    
    if args.smoke_test:
        args.windows = "0,4096"
        args.densities = "0.05"
        args.models = "qwen2,tinyllama"
        args.n_seeds = "1"
        args.out_csv = "scripts/benchmark/smoke_results_yarn_64.csv" if args.rope_type == "yarn" else "scripts/benchmark/smoke_results_64.csv"
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IS_CO = os.path.exists("/code") or os.path.exists("/data") or os.path.exists("/results")

    # Ensure no spaces and split correctly
    models = [m.strip() for m in args.models.split(',')]
    windows = [int(w) for w in args.windows.split(',')]
    densities = [float(d) for d in args.densities.split(',')]
    seeds_list = [int(s) for s in args.n_seeds.split(',')]

    # Resolve qa_dataset path
    qa_path = args.qa_dataset
    if not os.path.exists(qa_path):
        qa_path = os.path.join(BASE_DIR, os.path.basename(qa_path))
        if not os.path.exists(qa_path):
            qa_path = os.path.join(BASE_DIR, "qa_dataset.json")

    # Resolve out_csv path
    out_csv = args.out_csv
    if IS_CO:
        out_csv = os.path.join("/results", os.path.basename(out_csv))
    else:
        if not os.path.isabs(out_csv):
            out_dir = os.path.dirname(out_csv)
            if out_dir and not os.path.exists(out_dir):
                out_csv = os.path.join(BASE_DIR, os.path.basename(out_csv))

    if args.smoke_test:
        args.windows = "0,4096"
        args.densities = "0.05"
        args.models = "qwen2,tinyllama"
        args.n_seeds = "1"
        smoke_file = "smoke_results_yarn_64.csv" if args.rope_type == "yarn" else "smoke_results_64.csv"
        out_csv = os.path.join("/results" if IS_CO else BASE_DIR, smoke_file)

    # 1. Load context
    print("\n[INIT] Loading Context Files...")
    full_context = ""
    for f in args.context_files.split(','):
        f = f.strip()
        full_path = f if os.path.exists(f) else os.path.join(ARTICLES_DIR, os.path.basename(f))
        if os.path.exists(full_path):
            with open(full_path, 'r', encoding='utf-8') as fh:
                full_context += "\n\n" + fh.read()
        else:
            print(f"File not found: {f} (also tried {full_path})")
            
    # 2. Load QA data
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    if args.smoke_test:
        qa_data = qa_data[:5] # Test only first 5 in smoke test
        print("[SMOKE TEST] Reduced to 5 questions.")

    # 3. Setup CSV Logging
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["Timestamp", "Model", "RoPE_Type", "RIS_Mode", "Window", "Density", "Seeds", "Question_ID", 
                 "Expected", "Predicted", "Correct", "Prob_Winner", "Pos_Limit_Met"]
                 
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    out_file = open(out_csv, 'a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        csv_writer.writeheader()
        
    print(f"\n[INIT] Grid Sweep: {len(models)} models x {len(windows)} windows x {len(densities)} densities x {len(seeds_list)} ensembles")

    # 4. Main Sweep Loop
    for m in models:
        # Load weights once per model
        current_loaded_model = None
        current_tokenizer = None
        current_max_len_loaded = None
        current_density_loaded = None
        current_seed_loaded = None
        
        for w in windows:
            for d in densities:
                for s_count in seeds_list:
                    # Skip density/seed variations if it's the baseline (w==0), we only need 1 run
                    if w == 0 and (d != densities[0] or s_count != seeds_list[0]):
                        continue

                    native_limit = 2048 if m == 'tinyllama' else 32768
                    if args.rope_type == "yarn" and w > 0 and w <= native_limit:
                        print(f"\n[SKIPPING] {m} at Window={w}. Native limit is {native_limit}. Testing extrapolation only.")
                        continue



                    mode_name = "BASELINE (Zero Context)" if w == 0 else f"Window={w} | Density={d} | Seeds={s_count}"
                    if IS_CO and m == "qwen2" and w == 0:
                        print("\n==========================================================================")
                        print("[NOTICE] Code Ocean Free Quota Optimizations")
                        print("To stay within the Code Ocean free compute limits (e.g., 1 hour quota),")
                        print("this reproducible capsule is configured by default to run a brief benchmark:")
                        print("  - Models: qwen2 (1.5B)")
                        print("  - Windows: 0 (Baseline) and 16384 (RIS-Kernel Stochastic)")
                        print("  - Density: 5% | Seeds: 1\n")
                        print("This rapid run allows validating that the inference pipeline executes successfully")
                        print("and generates the primary manuscript plots in under 15 minutes.\n")
                        print("Note: The plotting scripts dynamically merge the freshly computed points")
                        print("with pre-calculated results uploaded in the repository (e.g., for 32k, 64k,")
                        print("and multi-seed sweeps) to display complete and accurate curves.\n")
                        print("You can customize the sweep windows and models to evaluate different configs")
                        print("by editing the arguments in the /code/run script.")
                        print("==========================================================================")
                    print(f"\n{'='*50}")
                    print(f"SWEEP CONF: Model={m} | {mode_name} | RoPE={args.rope_type}")
                    
                    print(f"[RELOAD] Building Topology...")
                    # Cleanup previous
                    if current_loaded_model:
                        del current_loaded_model
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    model, tokenizer, _ = load_ris_model(
                        max_length=w if w > 0 else 32768,
                        selected_dtype="float32",
                        model_class=m,
                        density=d if w > 0 else 1.0, # Baseline gets dense attention
                        apply_rope=True, # Always apply ROPE to handle potential question length overflows
                        rope_type=args.rope_type,
                        n_seeds=s_count if w > 0 else 1,
                        local_window=1024 if w > 1024 else (w // 4 if w > 0 else 1024),
                        ris_mode=args.ris_mode,
                        b_max=args.b_max,
                        bypass_generation_map=True,
                        disable_ris=(w == 0)
                    )
                    current_loaded_model = model
                    current_tokenizer = tokenizer
                    
                    # 4.1 Prefill KV Cache
                    import gc; gc.collect()
                    
                    if w == 0:
                        full_text = "You are a scientific expert. Answer the following multiple-choice question exactly with one single letter.\n"
                        max_len_for_prefill = 32768
                    else:
                        sys_prompt = "You are a scientific expert. Read the following text and answer the multiple-choice question exactly with one letter.\n\nTEXT:\n"
                        full_text = sys_prompt + full_context
                        max_len_for_prefill = w
                    
                    base_pkv, allocated_tokens = chunked_prefill(model, tokenizer, full_text, max_len=max_len_for_prefill)
                    opt_tokens = get_option_token_ids(tokenizer)
                    
                    # 4.2 Evaluate Questions
                    correct_count = 0
                    valid_qs = 0
                    
                    print(f"[EVALUATING] Running QA against base cache (Allocated: {allocated_tokens} tokens)")
                    for qa in tqdm(qa_data, desc="QA Eval"):
                        q_pos = qa["answer_token_position"]
                        
                        # Anti-Blindness Check
                        if w == 0:
                            pos_met = False
                            # In baseline, we evaluate all questions to get the random/pretrained baseline accuracy
                            evaluate_question = True
                        else:
                            pos_met = q_pos < (allocated_tokens * 0.85)
                            evaluate_question = pos_met or args.smoke_test
                        
                        if not evaluate_question:
                            continue
                            
                        valid_qs += 1
                        
                        # Format Discriminative Prompt
                        # Force attention to text and away from lazy "E" answers
                        prompt = f"\n\nQuestion: {qa['question']}\n\nOptions:\n"
                        for opt in qa['options']:
                            prompt += f"{opt}\n"
                        prompt += "\nRead the text carefully. If the information is present in the text, select the specific option. Only choose E if the information is absolutely missing.\nAnswer: "
                        
                        # Forward pass
                        predicted, probs = extract_qa_logits(model, tokenizer, prompt, base_pkv, opt_tokens)
                        
                        expected = qa['correct_option']
                        is_correct = 1 if predicted == expected else 0
                        correct_count += is_correct
                        
                        csv_writer.writerow({
                            "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "Model": m,
                            "RoPE_Type": args.rope_type,
                            "RIS_Mode": args.ris_mode,
                            "Window": w,
                            "Density": d,
                            "Seeds": s_count,
                            "Question_ID": qa["id"],
                            "Expected": expected,
                            "Predicted": predicted,
                            "Correct": is_correct,
                            "Prob_Winner": f"{probs[predicted]:.4f}",
                            "Pos_Limit_Met": pos_met
                        })
                        out_file.flush()
                        
                    accuracy = (correct_count / valid_qs) * 100 if valid_qs > 0 else 0
                    print(f"\n[METRIC] {correct_count}/{valid_qs} correct -> {accuracy:.2f}% Accuracy")

    out_file.close()
    print(f"\n[DONE] Benchmark saved to {out_csv}")

if __name__ == "__main__":
    main()
