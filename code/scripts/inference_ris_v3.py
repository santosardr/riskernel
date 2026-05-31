import torch
import os
import re
import argparse
import sys
import warnings
import itertools
import threading
import time
import readline
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList, logging as transformers_logging
from peft import PeftModel
from tqdm import tqdm
from ris_attention import replace_attention_with_ris

# Memory Bus Optimization (V9.1)
if "--threads" in sys.argv:
    try:
        idx = sys.argv.index("--threads")
        num_t = int(sys.argv[idx+1])
        torch.set_num_threads(num_t)
        torch.set_num_interop_threads(num_t)
    except: pass

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# BASE_DIR and environment detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISKERNEL_ROOT = os.path.dirname(BASE_DIR)

# BASE CONFIGURATION
TINYLLAMA_BASE_DIR = os.path.join(RISKERNEL_ROOT, "models")
QWEN2_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
ORIGINAL_TINYLLAMA_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAINING_MODELS_DIR = os.path.join(RISKERNEL_ROOT, "models")

# CODE OCEAN (CO) ENVIRONMENT DETECTION
IS_CO = os.path.exists("/code") or os.path.exists("/data") or os.path.exists("/results")
if IS_CO:
    TINYLLAMA_BASE_DIR = "/data/models"
    TRAINING_MODELS_DIR = "/data/models"
    ARTICLES_DIR = "/data/articles" if os.path.exists("/data/articles") else "/data"
else:
    ARTICLES_DIR = os.path.join(BASE_DIR, "data")



def warn_consistency(message):
    print(f"[CONSISTENCY WARNING] {message}")


def _extract_version_from_name(name):
    match = re.search(r"-v(\d+)$", name)
    return int(match.group(1)) if match else None


def find_latest_base_model(base_dir, model_prefix, pureris=True):
    """Returns the most recent merged base model that respects the specified architecture."""
    if not os.path.exists(base_dir):
        return None

    arch_tag = "pureris_" if pureris else "bigbird_"
    # Regex accepting optional prefixes (pureris_ or bigbird_) and capturing the version
    pattern = re.compile(rf"^(?:pureris_|bigbird_)?{re.escape(model_prefix)}-v(\d+)$")
    
    candidates = []
    for item in os.listdir(base_dir):
        full_path = os.path.normpath(os.path.join(base_dir, item))
        if not os.path.isdir(full_path):
            continue
            
        match = pattern.match(item)
        if match:
            # Filtro estrito: se o usuário pediu pureris, ignoramos QUALQUER pasta que comece com bigbird_
            # E se pediu bigbird, ignoramos qualquer pasta que comece com pureris_
            other_tag = "bigbird_" if pureris else "pureris_"
            if item.startswith(other_tag):
                continue
                
            version = int(match.group(1))
            # Score criteria: Version (weight 10) + Architecture Match (weight 2) + Legacy (weight 1)
            is_match = item.startswith(arch_tag)
            is_legacy = not (item.startswith("pureris_") or item.startswith("bigbird_"))
            
            score = version * 10
            if is_match: score += 2  # Exact arch match wins
            if is_legacy: score += 1 # Legacy wins over incorrect arch
            
            candidates.append((score, full_path))

    if not candidates:
        return None

    # Ordenação por score decrescente
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_training_run(models_dir, prefix, max_length):
    """Returns the most recent training directory following {prefix}ris_finetuned_vN_{window}."""
    if not os.path.exists(models_dir):
        return None, None

    # Strict Regex: requires pureris_ or bigbird_ at the folder start
    pattern = re.compile(rf"^(?:pureris_|bigbird_){re.escape(prefix)}finetuned_v(\d+)_{max_length}$")
    candidates = []
    for item in os.listdir(models_dir):
        full_path = os.path.join(models_dir, item)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(item)
        if match:
            candidates.append((int(match.group(1)), full_path))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    return candidates[0]


def infer_model_version_from_path(model_path):
    return _extract_version_from_name(os.path.basename(str(model_path).rstrip('/')))


def infer_model_family_from_path(model_path):
    model_lower = str(model_path).lower()
    if 'qwen' in model_lower:
        return 'qwen2'
    if 'tinyllama' in model_lower:
        return 'tinyllama'
    return None


def resolve_base_model_for_version(model_class, version, pureris=True):
    """Resolves the correct base model for a specific training version."""
    arch_tag = "pureris_" if pureris else "bigbird_"
    base_name = "TinyLlama-Bio" if model_class == 'tinyllama' else "Qwen-Bio"
    
    # 1. Look for base with the same architecture prefix (Recommended)
    prefixed_model = os.path.join(TINYLLAMA_BASE_DIR, f"{arch_tag}{base_name}-v{version}")
    if os.path.isdir(prefixed_model):
        return prefixed_model, f"{arch_tag}{base_name}-v{version}"
    
    # 2. Fallback to base without prefix (Legacy)
    legacy_model = os.path.join(TINYLLAMA_BASE_DIR, f"{base_name}-v{version}")
    if os.path.isdir(legacy_model):
        return legacy_model, f"{base_name}-v{version}"

    if model_class == 'tinyllama':
        return ORIGINAL_TINYLLAMA_MODEL_ID, "TinyLlama original"

    return QWEN2_MODEL_ID, "Qwen2-1.5B-Instruct"


def resolve_latest_base_only_model(model_class, pureris=True):
    """Resolves the most recent merged model for use with --base_only, respecting arch."""
    if model_class == 'tinyllama':
        latest_model = find_latest_base_model(TINYLLAMA_BASE_DIR, "TinyLlama-Bio", pureris=pureris)
        if latest_model:
            return latest_model, os.path.basename(latest_model)
        return ORIGINAL_TINYLLAMA_MODEL_ID, "TinyLlama original"

    latest_model = find_latest_base_model(TINYLLAMA_BASE_DIR, "Qwen-Bio", pureris=pureris)
    if latest_model:
        return latest_model, os.path.basename(latest_model)
    return QWEN2_MODEL_ID, "Qwen2-1.5B-Instruct"

class Spinner:
    def __init__(self, message="Processing"):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.busy = False
        self.delay = 0.1
        self.message = message
        self.thread = None
    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
    def _spin(self):
        while self.busy:
            self.write(f"\r{next(self.spinner)} {self.message}")
            time.sleep(self.delay)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        time.sleep(self.delay)
        if self.thread:
            self.thread.join()


def load_ris_model(max_length=65536, selected_dtype="float32", model_class='qwen2', density=0.05, base_model_path=None, pure_ris=False, apply_rope=True, rope_type="linear", seed=42, local_window=1024, n_seeds=1, needs_graph=False, global_window=256, b_max=2048, ris_mode='stochastic', disable_ris=False, bypass_generation_map=False):
    from ris_attention import replace_attention_with_ris
    from transformers import AutoConfig
    
    if model_class == 'qwen2':
        model_id = "Qwen/Qwen2-1.5B-Instruct" if not base_model_path else base_model_path
    else:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" if not base_model_path else base_model_path
    
    print(f"[1/4] Loading Tokenizer... ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    
    print(f"[2/4] Configuring RoPE Automation...")
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    if selected_dtype == "float32":
        current_dtype = torch.float32
    elif selected_dtype == "bfloat16":
        current_dtype = torch.bfloat16
    else:
        current_dtype = torch.float16
    
    native_limit = getattr(config, "max_position_embeddings", 2048 if model_class == 'tinyllama' else 32768)
    if apply_rope and max_length and max_length > native_limit:
        rope_factor = max_length / native_limit
        print(f"--- WARNING: Applying RoPE Scaling ({rope_type}, factor {rope_factor:.2f}) for {max_length} tokens ---")
        if model_class == 'qwen2':
            if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
                config.rope_parameters = {"rope_theta": 1000000.0, "rope_type": "default"}
            config.rope_parameters["rope_type"] = rope_type
            config.rope_parameters["factor"] = float(rope_factor)
            config.max_position_embeddings = max_length
        else:
            if not hasattr(config, "rope_theta") or config.rope_theta is None:
                config.rope_theta = 10000.0
            if rope_type == "yarn":
                import math
                attention_factor = 0.1 * math.log(rope_factor) + 1.0
                config.rope_scaling = {"rope_type": rope_type, "factor": float(rope_factor), "original_max_position_embeddings": native_limit, "attention_factor": attention_factor}
            else:
                config.rope_scaling = {"rope_type": rope_type, "factor": float(rope_factor), "original_max_position_embeddings": native_limit}
            config.max_position_embeddings = max_length
    elif max_length and max_length > native_limit:
        print(f"--- INFO: Running with Native RIS Mapping for {max_length} tokens without explicit RoPE ---")
        config.max_position_embeddings = max_length

    print(f"[3/4] Loading Base Weights in {current_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=current_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    )

    if disable_ris:
        print("[4/4] RIS Bypassed (Running Native Dense Attention)...")
    else:
        print(f"[4/4] Injecting RIS O(N) Geometry (Density: {density*100}% | Seeds: {n_seeds})...")
        base_model = replace_attention_with_ris(
            base_model, 
            seq_len=max_length, 
            density=density, 
            local_window=local_window, 
            global_window=global_window,
            seed=seed, 
            n_seeds=n_seeds,
            needs_graph=needs_graph,
            b_max=b_max,
            ris_mode=ris_mode,
            bypass_generation_map=bypass_generation_map
        )
    base_model.eval()

    return base_model, tokenizer, "V3-MacroRAG"

def _pearson_correlation(text1, text2):
    import re, math
    from collections import Counter
    words1 = [w for w in re.findall(r'\b\w+\b', text1.lower()) if len(w) > 3]
    words2 = [w for w in re.findall(r'\b\w+\b', text2.lower()) if len(w) > 3]
    
    if not words1 or not words2:
        return 0.0
        
    vocab = set(words1).union(set(words2))
    c1 = Counter(words1)
    c2 = Counter(words2)
    
    v1 = [c1[w] for w in vocab]
    v2 = [c2[w] for w in vocab]
    
    n = len(vocab)
    if n == 0: return 0.0
    
    mean1 = sum(v1) / n
    mean2 = sum(v2) / n
    
    num = sum((x - mean1) * (y - mean2) for x, y in zip(v1, v2))
    den = math.sqrt(sum((x - mean1)**2 for x in v1) * sum((y - mean2)**2 for y in v2))
    
    if den == 0:
        return 0.0
    return num / den


def apply_semantic_truncation(question, response, threshold=0.0, window_size=4):
    import re
    parts = re.split(r'(?<=[.!?\n])\s+', response.strip())
    sentences = [p for p in parts if p.strip()]
    
    if len(sentences) <= window_size:
        return response
        
    seed_text = question + " " + " ".join(sentences[:window_size])
    valid_sentences = sentences[:window_size]
    
    for i in range(window_size, len(sentences)):
        current_sentence = sentences[i]
        
        # Mandatory stop on Markdown headers (Structural drift)
        if re.search(r'^#{1,6}\s+', current_sentence, flags=re.MULTILINE):
            break

        # Keep short sentences (connectors)
        if len(current_sentence.split()) < 4:
            valid_sentences.append(current_sentence)
            continue
            
        corr = _pearson_correlation(seed_text, current_sentence)
        if corr >= threshold:
            valid_sentences.append(current_sentence)
        # else: we skip this sentence but continue searching the rest of the text
        
    return " ".join(valid_sentences)


def _sanitize_response(text):
    """Post-processing: remove leaked markdown/HTML artifacts."""
    import re as _re
    text = _re.sub(r'<[^>]+>', '', text)
    text = _re.sub(r'^#{1,6}\s+.*$', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'\|[-:]+\|', '', text)
    text = _re.sub(r'^\|.*\|$', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'\[@[^\]]+\]', '', text)
    text = _re.sub(r'\{#[^}]+\}', '', text)
    text = _re.sub(r'--- SOURCE:.*?---', '', text)
    text = _re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class SemanticStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, question, threshold=0.1, bootstrap_count=5):
        self.tokenizer = tokenizer
        self.question = question
        self.threshold = threshold
        self.bootstrap_count = bootstrap_count
        self.seed_sentences = []
        self.current_sentence_tokens = []
        self.sentence_count = 0
        self.stop_triggered = False
        self.validated_sentences = []
        
        # Guardrails: Patterns indicating drift toward "Instructional/Chat" mode
        self.hallucination_patterns = [
            "please write", "detailed explanation", "i hope this", 
            "could you", "can you", "write a", "explain to",
            "if you need", "don't forget", "bullet points"
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.stop_triggered: return True
        
        # Get last token
        last_token = input_ids[0, -1].item()
        self.current_sentence_tokens.append(last_token)
        
        text = self.tokenizer.decode(self.current_sentence_tokens, skip_special_tokens=True)
        
        # Mordaça em tempo real (não espera o ponto final se detectar padrão óbvio)
        text_lower = text.lower()
        if any(p in text_lower for p in self.hallucination_patterns):
            self.stop_triggered = True
            return True

        if text.endswith(('\n', '.', '!', '?')):
            sentence_to_test = text.strip()
            if not sentence_to_test: return False
            
            # Markdown header stop (Drift estrutural)
            import re
            if re.search(r'^#{1,6}\s+', sentence_to_test, flags=re.MULTILINE):
                self.stop_triggered = True
                return True

            # Fase 1: Buffer Seguro (Bootstrap)
            if self.sentence_count < self.bootstrap_count:
                self.seed_sentences.append(sentence_to_test)
                self.validated_sentences.append(sentence_to_test)
                self.sentence_count += 1
                self.current_sentence_tokens = []
                return False

            # Fase 2: Análise de Pearson (A partir da 6ª frase)
            seed_text = self.question + " " + " ".join(self.seed_sentences)
            corr = _pearson_correlation(seed_text, sentence_to_test)
            
            if corr < self.threshold:
                self.stop_triggered = True
                return True
            
            self.validated_sentences.append(sentence_to_test)
            self.sentence_count += 1
            self.current_sentence_tokens = []
        
        return False

class DynamicSemanticStreamer(TextStreamer):
    """
    Experimental RIS-Filtering Streamer controlled by SemanticStoppingCriteria.
    Features a cursor-level spinner for visual feedback during buffering.
    """
    def __init__(self, tokenizer, criteria, skip_prompt=True, skip_special_tokens=True):
        # Initial config
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.criteria = criteria
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.printed_sentences_count = 0
        
        # Cursor Spinner state
        import itertools as _it
        import threading as _th
        self.spinner_chars = _it.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.spinner_running = False
        self.stop_spinner_event = _th.Event()
        self.spinner_thread = None
        self._start_spinner()

    def _spinner_loop(self):
        import time as _time
        import sys as _sys
        while not self.stop_spinner_event.is_set():
            _sys.stdout.write(next(self.spinner_chars))
            _sys.stdout.flush()
            _time.sleep(0.1)
            _sys.stdout.write('\b \b') # Erase spinner
            _sys.stdout.flush()

    def _start_spinner(self):
        import threading as _th
        if not self.spinner_running:
            self.stop_spinner_event.clear()
            self.spinner_thread = _th.Thread(target=self._spinner_loop, daemon=True)
            self.spinner_thread.start()
            self.spinner_running = True

    def _stop_spinner(self):
        if self.spinner_running:
            self.stop_spinner_event.set()
            if self.spinner_thread:
                self.spinner_thread.join(timeout=0.2)
            self.spinner_running = False

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Check if criteria has new validated sentences to print
        if len(self.criteria.validated_sentences) > self.printed_sentences_count:
            # Temporarily stop spinner to print the real content
            self._stop_spinner()
            
            for i in range(self.printed_sentences_count, len(self.criteria.validated_sentences)):
                s = self.criteria.validated_sentences[i]
                # Print connector space and sentence
                print(" " + s, end="", flush=True) 
            
            self.printed_sentences_count = len(self.criteria.validated_sentences)
            
            # Resume spinner for next buffering
            self._start_spinner()

    def end(self):
        # Print leftover tokens in the criteria's buffer (Crucial for code ending without \n)
        if self.criteria.current_sentence_tokens:
            last_text = self.tokenizer.decode(self.criteria.current_sentence_tokens, skip_special_tokens=True).strip()
            if last_text:
                print(" " + last_text, end="", flush=True)
                self.criteria.validated_sentences.append(last_text)
        self._stop_spinner()


from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0, -1] == stop_id:
                return True
        return False

def get_context_hash(context_text, window, model_class, n_seeds, density, b_max, ris_mode):
    """Generates two hashes: one for the base (context) and another for the RIS geometry."""
    import hashlib
    # Base Hash: Depends only on content and model
    base_payload = f"{context_text[:1000]}_{context_text[-1000:]}_{window}_{model_class}"
    base_hash = hashlib.sha256(base_payload.encode()).hexdigest()[:16]
    
    # RIS Hash: Depends on base + stochastic parameters
    ris_payload = f"{base_hash}_{n_seeds}_{density}_{b_max}_{ris_mode}"
    ris_hash = hashlib.sha256(ris_payload.encode()).hexdigest()[:16]
    
    return base_hash, ris_hash

def chat_loop(model, tokenizer, total_max_length, chat_buffer_size, initial_context="", temp=0.1, top_k=50, top_p=0.9, version="V5.9-Ultimate", repetition_penalty=1.3, semantic_filter=False, semantic_threshold=0.1, model_class='tinyllama', n_seeds=1, density=0.01, b_max=2048, ris_mode='stochastic'):
    print("\n" + "="*60)
    print(f" RIS-Kernel {version} | Base: dynamic | Window: {total_max_length} Tokens")
    print(f" [Knowledge: {total_max_length - chat_buffer_size} | Chat: {chat_buffer_size} buffer]")
    print(f" (Biochemistry Specialist | Fortified Coherence Active)")
    print("="*60 + "\n")

    # --- PERSISTENCE SYSTEM (Anti-Frustration V2: Dual-Hash) ---
    if IS_CO:
        cache_dir = "/tmp/ris_cache"
    else:
        cache_dir = os.path.join(BASE_DIR, "data", "ris_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    base_hash, ris_hash = get_context_hash(initial_context, total_max_length, model_class, n_seeds, density, b_max, ris_mode)
    
    # File Mapping
    cache_path = os.path.join(cache_dir, f"{model_class}_cache_{base_hash}.pt")
    ris_cache_path = os.path.join(cache_dir, f"{model_class}_ris_{ris_hash}.pt")
    
    past_key_values = None
    curr_len = 0
    loaded_from_cache = False

    from ris_attention import load_ris_state, save_ris_state

    # [RESCUE]: Prefill skip if 3.6GB KV-Cache exists
    if os.path.exists(cache_path):
        print(f"[SYSTEM] Base Memory detected: {base_hash}. Restoring state...")
        t_load = time.time()
        try:
            past_key_values = torch.load(cache_path, weights_only=False)
            # Try to load specific geometry for these seeds
            if os.path.exists(ris_cache_path):
                load_ris_state(ris_cache_path, device=model.device, density=density, b_max=b_max, ris_mode=ris_mode)
            else:
                print(f"[SYSTEM] New RIS configuration ({n_seeds} seeds). Activating fast reconstruction...")
            
            loaded_from_cache = True
            print(f"[SYSTEM] Brain restored in {time.time() - t_load:.1f}s. Skipping prefill.")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}. Starting standard prefill...")

    # ===========================================================
    # SYSTEM PROMPT — Anti-Hallucination Guardrails (V2 / Phase 3)
    # ===========================================================
    SYSTEM_PROMPT = (
        f"You are RIS-Kernel {version}, a high-level technical Oracle specializing in biochemistry and molecular biology. "
        "Your mission is to provide dense, technically rigorous responses without conversational filler, preambles, or social etiquette. "
        "Incorporate specific kinetic parameters (Km, Vmax, kcat), thermodynamic data (deltaG, deltaS, enthalpy), and molecular structural details into every explanation. "
        "Focus on concrete biochemical strategies over abstract generalizations. "
        "If requested for computational tools or code, provide the code implementation directly without introductory text."
    )

    # 1. PREFILL SYSTEM PROMPT (Hybrid Architecture V4.7)
    if model_class == 'qwen2':
        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nKNOWLEDGE BASE:\n{initial_context}" if initial_context else SYSTEM_PROMPT}]
        system_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        system_text = f"[SYSTEM]: {SYSTEM_PROMPT}\n"
        if initial_context:
            system_text += f"<KNOWLEDGE_BASE>\n{initial_context}\n</KNOWLEDGE_BASE>\n"
        system_text += "Technical Analysis Segment:\n"
    
    inputs = tokenizer(system_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    total_tokens = inputs['input_ids'].shape[1]
    curr_len = total_tokens

    if not loaded_from_cache:
        # --- CHUNKED PREFILL (V5.2: Prevents 122GB RAM spike) ---
        all_input_ids = inputs['input_ids']
        chunk_size = 1024
        past_key_values = None
        
        print(f"\n[SYSTEM] Initializing biochemical brain ({total_tokens} tokens)...")
        print(f"[SYSTEM] Processing in chunks of {chunk_size} to save RAM...")
        
        from tqdm import tqdm
        t_start = time.time()
        
        with torch.no_grad():
            for i in range(0, total_tokens, chunk_size):
                end_idx = min(i + chunk_size, total_tokens)
                chunk = all_input_ids[:, i:end_idx]
                outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                if i == 0:
                    pbar = tqdm(total=total_tokens, desc="[SYSTEM] Ingestion", unit="tk")
                pbar.update(chunk.shape[1])
                
        pbar.close()
        t_end = time.time()
        print(f"[SYSTEM] Base KV-Cache consolidated in {t_end - t_start:.1f}s ({total_tokens / max(1, t_end - t_start):.1f} tokens/s).")
        
        # SAVE CACHE FOR NEXT SESSION (Anti-Crash Version)
        print(f"[SYSTEM] Saving persistent brain in {cache_path}...")
        torch.save(past_key_values, cache_path)
        save_ris_state(ris_cache_path)

    # CHECKPOINT V5.0: Saves document "snapshot" for instant rollback
    base_past_key_values = past_key_values 
    base_curr_len = curr_len
    
    # WARM-UP V5.5: Triggers C++ compilation (JIT) before first question
    print("[SYSTEM] Warming up response engine (C++ JIT Compilation)...")
    with torch.no_grad():
        warmup_input = torch.tensor([[tokenizer.eos_token_id]], device=model.device)
        _ = model(warmup_input, past_key_values=base_past_key_values, use_cache=True)
    print("[SYSTEM] Checkpoint created. Engine ready and optimized.\n")

    # Initialize log path
    log_path = getattr(model, "log_path", "inference_sessions.log")
    with open(log_path, "a", encoding="utf-8") as f_log:
        f_log.write(f"\n{'='*20} SESSION: {time.strftime('%Y-%m-%d %H:%M:%S')} {'='*20}\n")

    while True:
        # Pre-turn state backup for error recovery (Ctrl+C safety)
        backup_pkv = past_key_values
        backup_curr_len = curr_len

        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input.strip(): continue

            # COMMAND: /reset (Clears context memory)
            if user_input.lower() == '/reset':
                print("[SYSTEM]: Chat memory reset (Rollback to Documents).")
                past_key_values = base_past_key_values
                curr_len = base_curr_len
                print(f"[SYSTEM] Instant rollback complete.\n")
                continue

            # 2. APPEND USER INPUT
            if model_class == 'qwen2':
                prefix = "<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n"
                user_text = f"{prefix}{user_input}{suffix}"
            else:
                user_text = f"\nBiochemist's Question: {user_input}\nScientific Oracle Response: "

            # --- MEMORY CLEANUP (V6.5) ---
            from ris_attention import clear_coordination_cache
            clear_coordination_cache()

            user_inputs = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            input_ids = user_inputs['input_ids']
            
            batch_size = input_ids.shape[0]
            new_len = input_ids.shape[1]
            total_len = curr_len + new_len

            if total_len > total_max_length:
                print(f"[System]: Window full ({total_len} > {total_max_length}).")
                print(f"[System]: Executing Automatic Rollback to Knowledge Base...")
                past_key_values = base_past_key_values
                curr_len = base_curr_len
                
                # Try re-tokenizing only the current question
                user_inputs = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).to(model.device)
                input_ids = user_inputs['input_ids']
                total_len = curr_len + input_ids.shape[1]
                
                if total_len > total_max_length:
                    print("[CRITICAL FAILURE]: Question is too long even for empty cache. Truncating.")
                    continue

            # MOLD AND POSITIONS (Essential to avoid breaking the KV-Cache)
            position_ids = torch.arange(curr_len, total_len, device=model.device).unsqueeze(0)
            attention_mask = torch.ones((batch_size, total_len), device=model.device, dtype=torch.long)

            # Stop Criteria (Guardrails V6.0)
            stop_ids = [tokenizer.eos_token_id]
            if model_class == 'qwen2':
                stop_ids.extend([151644, 151645])
            
            # 1. ID-based Stop (Basic)
            id_stop = StopOnTokens(stop_ids)
            
            # 2. Semantic and Pattern Stop (Pearson + Anti-Draft)
            semantic_stop = SemanticStoppingCriteria(
                tokenizer, 
                user_input, 
                threshold=semantic_threshold if semantic_filter else -1.0,
                bootstrap_count=10
            )
            
            stopping_criteria = StoppingCriteriaList([id_stop, semantic_stop])

            print(f"Response: ", end="", flush=True)
            
            from transformers import TextStreamer
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.no_grad():
                generate_outputs = model.generate(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    streamer=streamer,
                    return_dict_in_generate=True
                )

            # Response Extraction and Metrics
            gen_ids = generate_outputs.sequences[0]
            new_tokens_produced = gen_ids[input_ids.shape[1]:]
            response_text = tokenizer.decode(new_tokens_produced, skip_special_tokens=True).strip()
            
            # Safety Sanitization (In case stopping_criteria misses a token)
            for stop_term in ["Sure!", "I'm sorry", "Please note", "You:", "Biochemist"]:
                if stop_term in response_text:
                    response_text = response_text.split(stop_term)[0].strip()

            # Update cache for next turn
            # generate_outputs.past_key_values already contains full accumulated session cache.
            past_key_values = generate_outputs.past_key_values
            
            # Total sequence size in cache is the size of fully generated sequence
            if hasattr(past_key_values, "get_seq_length"):
                curr_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                curr_len = past_key_values[0][0].shape[-2]
            else:
                curr_len = total_len + new_tokens_produced.shape[0]

            # Logging
            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(f"USER: {user_input}\n")
                f_log.write(f"ASSISTANT: {response_text}\n")
                f_log.write(f"{'-'*40}\n")

            import gc; gc.collect()

        except KeyboardInterrupt:
            print("\n[SYSTEM]: Interruption detected. Restoring previous cache...")
            past_key_values = backup_pkv
            curr_len = backup_curr_len
            continue
        except Exception as e:
            print(f"\n[ERROR]: {str(e)}")
            past_key_values = backup_pkv
            curr_len = backup_curr_len
            break


def main():
    parser = argparse.ArgumentParser(description="RIS-Oracle V3: Massive Context Inference without LoRA for O(N) proof.")
    parser.add_argument("--window", type=int, default=32768, help="Reserved size for ARTICLES only (Content)")
    parser.add_argument("--chat_buffer", type=int, default=8192, help="Extra size reserved for CHAT (Default: 8192)")
    parser.add_argument("--temp", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--dtype", type=str, default="float32", help="Precision: float32 (Recommended for Haswell CPUs)")
    parser.add_argument("--context_files", type=str, default="", help="Comma-separated file list to inject. Ex: data/article.txt,data/lehninger.txt")
    parser.add_argument("--model_class", type=str, default='qwen2', help="Model family to use (qwen2/tinyllama)")
    parser.add_argument("--density", type=float, default=0.05, help="RIS mask density (default: 0.05)")
    parser.add_argument("--base_model_path", type=str, default=None, help="Custom path for the base model.")
    parser.add_argument("--pureris", action="store_true", default=False, help="Enable Pure RIS mode (stochastic). Disabled by default.")
    parser.add_argument("--bigbird", action="store_false", dest="pureris", help="Enable BigBird Hybrid mode (Default)")
    parser.add_argument("--apply_rope", action="store_true", help="Apply RoPE Scaling if max_length > native limit")
    parser.add_argument("--seed", type=int, default=42, help="RIS stochastic seed")
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--no_semantic_filter", action="store_false", dest="semantic_filter")
    parser.add_argument("--semantic_threshold", type=float, default=-0.1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--log_file", type=str, default="inference_ris_v3.log")
    parser.add_argument("--local_window", type=int, default=16, help="Local anchor size (default: 16, recommended for RAG: 64)")
    parser.add_argument("--global_window", type=int, default=256, help="Absolute global anchor size (default: 256)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P Sampling (Nucleus)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K Sampling")
    parser.add_argument("--save_graph", type=str, default=None, help="Export stochastic attention graph (5%%) to a .dot file")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of independent RIS projections to merge (Ensemble)")
    parser.add_argument("--co", action="store_true", help="Enable Code Ocean compatibility mode (overrides resources)")
    parser.add_argument("--b_max", type=int, default=2048, help="Maximum block size for O(N) guarantee (structural mode)")
    parser.add_argument("--ris_mode", type=str, default='stochastic', choices=['stochastic', 'structural'],
                        help="RIS geometry mode: 'stochastic' (default, original) or 'structural' (clique decomposition)")

    args = parser.parse_args()

    # --- CODE OCEAN (CO) PREVALENCE LOGIC ---
    if args.co or IS_CO:
        print("\n" + "!"*60)
        print("[SYSTEM] CODE OCEAN (CO) ENVIRONMENT DETECTED OR FORCED")
        print("[CO] Overriding parameters for technical viability:")
        args.model_class = 'tinyllama'
        args.threads = 8 # Balanced spot suggested by Anderson to avoid bus contention
        if args.window > 4096:
            print(f"[CO] Reducing window from {args.window} to 4096 (2x TinyLlama native: 2048).")
            args.window = 4096
        
        # Stability Fix: Cap chat buffer in CO mode to keep RoPE factor manageable for 1B models
        if args.chat_buffer > 2048:
            print(f"[CO] Balancing chat buffer: {args.chat_buffer} -> 2048 (Coherence Guardrail)")
            args.chat_buffer = 2048
            
        # Ensemble configuration to ensure biochemical parameter recall
        print(f"[CO] Ensemble Active: Setting 4 seeds with 5.0% density.")
        args.density = 0.05
        args.n_seeds = 4
            
        args.dtype = "float32"
        args.apply_rope = True # Enable RoPE scaling by default for CO stability
        print("!"*60 + "\n")

    # --- AUTO-ROPE SCALING ACTIVATION ---
    total_capacity = args.window + args.chat_buffer
    if not args.apply_rope:
        if args.model_class == 'tinyllama' and total_capacity > 2048:
            print(f"[SYSTEM] Auto-RoPE: TinyLlama exceeds 2k native limit ({total_capacity}). Activating scaling...")
            args.apply_rope = True
        elif args.model_class == 'qwen2' and total_capacity > 32768:
            print(f"[SYSTEM] Auto-RoPE: Qwen2 exceeds 32k native limit ({total_capacity}). Activating scaling...")
            args.apply_rope = True

    if not hasattr(args, 'semantic_filter'):
        args.semantic_filter = True

    import psutil
    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count()
    infer_cores = args.threads if args.threads else min(physical_cores, 8)
    total_capacity = args.window + args.chat_buffer
    
    os.environ["OMP_NUM_THREADS"] = str(infer_cores)
    os.environ["MKL_NUM_THREADS"] = str(infer_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(infer_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(infer_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(infer_cores)
    torch.set_num_threads(infer_cores)
    print(f"--- RIS INFERENCE V3 (Using {infer_cores}/{physical_cores} physical CPUs) ---")

    torch.manual_seed(args.seed)

    model, tokenizer, active_version = load_ris_model(
        max_length=total_capacity, 
        selected_dtype=args.dtype, 
        model_class=args.model_class, 
        density=args.density, 
        base_model_path=args.base_model_path, 
        pure_ris=args.pureris, 
        apply_rope=args.apply_rope, 
        seed=args.seed,
        local_window=args.local_window,
        n_seeds=args.n_seeds,
        needs_graph=bool(args.save_graph),
        global_window=args.global_window,
        b_max=args.b_max,
        ris_mode=args.ris_mode
    )
    
    model.log_path = args.log_file
    model.adapter_path = "NO_ADAPTER_(V3_MACRO_RAG)"
    
    initial_context = ""
    if args.context_files:
        print("[SYSTEM] Processing massive context attachments...")
        raw_text = ""
        for filepath in args.context_files.split(','):
            filepath = filepath.strip()
            # Try relative path to ARTICLES_DIR if file not found in absolute path
            full_path = filepath if os.path.exists(filepath) else os.path.join(ARTICLES_DIR, os.path.basename(filepath))
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    doc_content = f.read()
                    raw_text += f"\n<DOCUMENT source=\"{os.path.basename(full_path)}\">\n{doc_content}\n</DOCUMENT>\n"
            else:
                print(f"[WARNING] Context file not found: {filepath} (also tried in {ARTICLES_DIR})")

        if raw_text:
            print("[SYSTEM] Tokenizing text to ensure window limit...")
            # Now safe_limit is EXACTLY what the user requested for the articles
            safe_limit = args.window
            tokens = tokenizer(raw_text, add_special_tokens=False)['input_ids']
            
            if len(tokens) > safe_limit:
                print(f"[INFO] Truncating context: {len(tokens)} -> {safe_limit} tokens to fit RIS window.")
                tokens = tokens[:safe_limit]
                raw_text = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                print(f"[INFO] Context fit fully: {len(tokens)} tokens.")
                
            # --- CONTEXT ECHO REINFORCEMENT (V5.7) ---
            initial_context = raw_text + "\n\n[REINFORCEMENT]: Base your response SOLELY on the documents above. Ignore previous knowledge if it diverges. Answer technically."

    if args.save_graph:
        from ris_attention import export_ris_brain_graph
        print(f"[SYSTEM] Preparing RIS Graph export (5%% factor of {total_capacity} tokens)...")
        # Re-build prompt exactly as chat_loop will to ensure correct node IDs
        SYSTEM_PROMPT = (
            "You are RIS-Kernel V5.9-Ultimate, a high-level technical Oracle specializing in biochemistry and molecular biology. "
            "Your mission is to provide dense, technically rigorous responses without conversational filler, preambles, or social etiquette. "
            "Incorporate specific kinetic parameters (Km, Vmax, kcat), thermodynamic data (deltaG, deltaS, enthalpy), and molecular structural details into every explanation. "
            "Focus on concrete biochemical strategies over abstract generalizations. "
            "If requested for computational tools or code, provide the code implementation directly without introductory text."
        )
        if args.model_class == 'qwen2':
            messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nBASE DE CONHECIMENTO:\n{initial_context}" if initial_context else SYSTEM_PROMPT}]
            system_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            system_text = "Biochemistry Technical Document: Lehninger Principles.\n"
            if initial_context:
                system_text += f"<KNOWLEDGE_BASE>\n{initial_context}\n</KNOWLEDGE_BASE>\n"
            system_text += "Technical Analysis Segment:\n"
        
        graph_tokens = tokenizer(system_text, add_special_tokens=False)['input_ids']
        export_ris_brain_graph(args.save_graph, tokenizer, graph_tokens)

    chat_loop(
        model, tokenizer, 
        total_max_length=total_capacity, 
        chat_buffer_size=args.chat_buffer, 
        initial_context=initial_context, 
        temp=args.temp, 
        top_k=args.top_k,
        top_p=args.top_p,
        version="V5.9-Ultimate", 
        repetition_penalty=args.repetition_penalty, 
        semantic_filter=args.semantic_filter, 
        semantic_threshold=args.semantic_threshold, 
        model_class=args.model_class, 
        n_seeds=args.n_seeds, 
        density=args.density,
        b_max=args.b_max,
        ris_mode=args.ris_mode
    )

if __name__ == "__main__":
    main()
