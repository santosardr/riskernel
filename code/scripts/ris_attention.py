import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import os
import transformers.models.llama.modeling_llama as modeling_llama
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

# RIS Global Settings
_RIS_CURRENT_SEED = 42
_RIS_LOCAL_WINDOW = 1024 
_RIS_GLOBAL_WINDOW = 256 

# MASTER STRUCTURES V6.2 (Ensemble Prefill + Fast Generation)
_MASTER_RIS_MASK = None      # Ensemble Mask for Prefill (SDPA)
_ACTIVE_INDEX_MAP = None     # Compact Map (1 seed) for Fast Generation
_ACTIVE_VALID_MASK = None    # Validity Mask for the Compact Map
_MASTER_STOCH_MAP = None     # Full Stochastic Map (Union, for export)

def _precompute_ris_structures(seq_len, n_seeds=1, density=0.01, w_local=1024, w_global=256, ris_seed=None, device='cuda', needs_graph=False, b_max=2048, ris_mode='stochastic', bypass_generation_map=False):
    """Generates the RIS Ensemble geometry. Mode: 'stochastic' (default) or 'structural'."""
    global _MASTER_RIS_MASK, _ACTIVE_INDEX_MAP, _ACTIVE_VALID_MASK, _MASTER_STOCH_MAP
    global _RIS_CURRENT_SEED, _RIS_LOCAL_WINDOW, _RIS_GLOBAL_WINDOW
    
    if ris_seed is not None:
        torch.manual_seed(ris_seed)
        _RIS_CURRENT_SEED = ris_seed
    
    _RIS_LOCAL_WINDOW = w_local
    _RIS_GLOBAL_WINDOW = w_global
    
    t0 = time.time()
    
    if ris_mode == 'structural':
        # [Structural RIS] Phase 2: Block size is driven by N, capped at b_max.
        # B_max acts as a hard upper limit to prevent memory overflows.
        block_size = max(1, min(int(seq_len * 0.1), b_max))
        # [Structural RIS] Phase 3: Global random sampling controlled by density.
        num_global = max(1, int(seq_len * density))
        num_stochastic = block_size + num_global
    else:
        # [Stochastic RIS] Pure global random sampling (original behavior).
        block_size = 0
        num_global = max(1, int(seq_len * density))
        num_stochastic = num_global
    
    # RAM Impact Estimation (Index Vector + Mask + Graph + Weights/KV)
    est_static_gb = 34.2 # Model (6.2GB) + KV-Cache float32 65k (28.0GB)
    total_w_compact = w_global + num_stochastic + 1
    est_idx_gb = (seq_len * n_seeds * total_w_compact * 5) / 1e9 # Index(4) + Valid(1)
    est_mask_gb = (seq_len * seq_len) / 1e9 if device == 'cpu' else 1.0
    est_graph_gb = (seq_len * n_seeds * num_stochastic * 4) / 1e9 if needs_graph else 0
    total_est = est_idx_gb + est_mask_gb + est_graph_gb + est_static_gb
    
    print(f"[RIS] Mode: {ris_mode.upper()} | Seeds: {n_seeds} | Window: {seq_len} tokens")
    if ris_mode == 'structural':
        block_pct = (block_size / seq_len) * 100
        global_pct = (num_global / seq_len) * 100
        print(f"[RIS]   Structural Clique: block_size = min(0.1*N={int(seq_len*0.1)}, B_max={b_max}) = {block_size} ({block_pct:.2f}% density)")
        print(f"[RIS]   Structural Global Redundancy: R = density*N = {num_global} ({global_pct:.2f}% density)")
    else:
        print(f"[RIS]   Stochastic: density={density*100:.1f}% => {num_stochastic} samples/token")
    print(f"[RIS] Systemic Estimation: {total_est:.2f} GB (RIS: {total_est-est_static_gb:.2f}GB | Static: {est_static_gb}GB)")
    
    if total_est > 115:
        print(f"[CRITICAL ALERT] Total occupancy of {total_est:.2f}GB is at the RAM limit (126GB).")
        print(f"[WARNING] Imminent SWAP risk. Suggestion: Lower density or reduce seeds.")

    # 1. Final Memory Allocation
    if bypass_generation_map:
        _ACTIVE_INDEX_MAP = None
        _ACTIVE_VALID_MASK = None
    else:
        _ACTIVE_INDEX_MAP = torch.zeros((seq_len, n_seeds, total_w_compact), dtype=torch.int32, device=device)
        _ACTIVE_VALID_MASK = torch.zeros((seq_len, n_seeds, total_w_compact), dtype=torch.bool, device=device)
    
    # [96GB OPTIMIZATION]: Only allocate graph map if exporting .dot
    if needs_graph:
        _MASTER_STOCH_MAP = torch.zeros((seq_len, n_seeds * num_stochastic), dtype=torch.int32, device=device)
    else:
        _MASTER_STOCH_MAP = None
    
    # 2. Initialize Global Mask (Prefill) with Local + Global Window
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    rows_vec = torch.arange(seq_len, device=device).view(-1, 1)
    
    # Local Sliding Window: row-by-row contiguous slice writing to avoid O(N^2) temporary int64 allocations (fixes OOM-Killer crashes)
    for i in range(seq_len):
        start_col = max(0, i - _RIS_LOCAL_WINDOW)
        mask[i, start_col : i + 1] = True
        
    mask[:, :_RIS_GLOBAL_WINDOW] = (rows_vec >= torch.arange(_RIS_GLOBAL_WINDOW, device=device))
    
    # Fill fixed indices in the generation map
    if not bypass_generation_map:
        g_range = torch.arange(_RIS_GLOBAL_WINDOW, device=device).view(1, 1, -1).to(torch.int32)
        _ACTIVE_INDEX_MAP[:, :, :_RIS_GLOBAL_WINDOW] = g_range
        _ACTIVE_VALID_MASK[:, :, :_RIS_GLOBAL_WINDOW] = (rows_vec.unsqueeze(1) >= g_range)
        
        _ACTIVE_INDEX_MAP[:, :, -1] = rows_vec.to(torch.int32)
        _ACTIVE_VALID_MASK[:, :, -1] = True

    # 3. STREAMING Processing (Seed by Seed to avoid 40GB spikes)
    rng = torch.Generator(device=device)
    row_indices = torch.arange(seq_len, device=device)
    
    for i in range(n_seeds):
        rng.manual_seed(_RIS_CURRENT_SEED + i)
        
        if ris_mode == 'structural':
            # --- Structural Mode: Stochastic Clique Decomposition ---
            # Randomly shuffles tokens, then partitions into dense blocks.
            # Block size = min(0.1*N, b_max) => O(N) cost guaranteed.
            perm = torch.randperm(seq_len, generator=rng, device=device).to(torch.int32)
            num_blocks = (seq_len + block_size - 1) // block_size
            padded_len = num_blocks * block_size
            padded_perm = torch.cat([perm, torch.full((padded_len - seq_len,), -1, dtype=torch.int32, device=device)])
            blocks = padded_perm.view(num_blocks, block_size)
            inv_perm = torch.zeros_like(perm)
            inv_perm[perm.long()] = torch.arange(seq_len, device=device, dtype=torch.int32)
            block_indices = (inv_perm // block_size).long()
            clique_map = blocks[block_indices]  # (seq_len, block_size)

            # --- Phase 3: Global Redundancy (controlled by density) ---
            global_map = torch.randint(0, seq_len, (seq_len, num_global), generator=rng, device=device).to(torch.int32)

            # Combine structural clique edges and global redundancy edges
            seed_map = torch.cat([clique_map, global_map], dim=1)  # (seq_len, num_stochastic)

            causal_filt = (seed_map <= row_indices.unsqueeze(1)) & (seed_map != -1)
            row_exp = row_indices.unsqueeze(1).expand(-1, num_stochastic)
            mask[row_exp[causal_filt], seed_map[causal_filt]] = True

            if not bypass_generation_map:
                valid_mask = (row_indices.unsqueeze(1).to(torch.int32) >= seed_map) & (seed_map != -1)
            del perm, inv_perm, blocks, clique_map, global_map
        else:
            # --- Stochastic Mode: Pure Global Random Sampling ---
            # Original behavior: uniform random edges based on density.
            seed_map = torch.randint(0, seq_len, (seq_len, num_stochastic), generator=rng, device=device).to(torch.int32)
            row_exp = row_indices.unsqueeze(1).expand(-1, num_stochastic)
            causal_filt = (seed_map <= row_exp)
            mask[row_exp[causal_filt], seed_map[causal_filt]] = True
            if not bypass_generation_map:
                valid_mask = (row_indices.unsqueeze(1).to(torch.int32) >= seed_map)
        
        # Store in DOT map and fast generation map
        c_start = i * num_stochastic
        if _MASTER_STOCH_MAP is not None:
            _MASTER_STOCH_MAP[:, c_start : c_start + num_stochastic] = seed_map
        if not bypass_generation_map:
            _ACTIVE_INDEX_MAP[:, i, _RIS_GLOBAL_WINDOW : _RIS_GLOBAL_WINDOW + num_stochastic] = seed_map
            _ACTIVE_VALID_MASK[:, i, _RIS_GLOBAL_WINDOW : _RIS_GLOBAL_WINDOW + num_stochastic] = valid_mask
        
        del seed_map
        if not bypass_generation_map:
            del valid_mask
        if i % 10 == 0: torch.cuda.empty_cache() if device == 'cuda' else None

    mask.fill_diagonal_(True)
    _MASTER_RIS_MASK = mask
    if _MASTER_STOCH_MAP is not None:
        _MASTER_STOCH_MAP = _MASTER_STOCH_MAP.cpu() # Keep in master RAM, free VRAM

    ens_density = mask.float().mean() * 100
    print(f"[RIS] Geometry synchronized in {time.time()-t0:.2f}s (Streaming: Zero Peak).")
    print(f"[RIS]   Recall: {ens_density:.2f}% (Ensemble {n_seeds} seeds)")
    return {'ens_density': ens_density, 'n_seeds': n_seeds}

def save_ris_state(checkpoint_path):
    """Saves only the parameters to reconstruct the RIS geometry."""
    global _RIS_LOCAL_WINDOW, _RIS_GLOBAL_WINDOW, _RIS_CURRENT_SEED, _ACTIVE_INDEX_MAP
    global _b_max_cache, _ris_mode_cache
    if _ACTIVE_INDEX_MAP is None: return False
    state = {
        'window': _ACTIVE_INDEX_MAP.shape[0],
        'n_seeds': _ACTIVE_INDEX_MAP.shape[1],
        'w_local': _RIS_LOCAL_WINDOW,
        'w_global': _RIS_GLOBAL_WINDOW,
        'ris_seed': _RIS_CURRENT_SEED,
        'b_max': _b_max_cache if '_b_max_cache' in globals() else 2048,
        'ris_mode': _ris_mode_cache if '_ris_mode_cache' in globals() else 'stochastic',
    }
    torch.save(state, checkpoint_path)
    print(f"[RIS] Geometry metadata saved in: {checkpoint_path}")

def load_ris_state(checkpoint_path, device='cuda', density=0.01, b_max=2048, ris_mode='stochastic'):
    """Loads parameters and RECONSTRUCTS the RIS geometry (Takes ~90s)."""
    if not os.path.exists(checkpoint_path): return False
    state = torch.load(checkpoint_path, weights_only=False)
    _precompute_ris_structures(
        seq_len=state['window'],
        n_seeds=state['n_seeds'],
        density=density,
        device=device,
        w_local=state.get('w_local', 1024),
        w_global=state.get('w_global', 256),
        ris_seed=state.get('ris_seed', 42),
        b_max=state.get('b_max', b_max),
        ris_mode=state.get('ris_mode', ris_mode)
    )
    return True

def export_ris_brain_graph(filename, tokenizer, input_ids):
    global _MASTER_RIS_MASK, _MASTER_STOCH_MAP, _RIS_LOCAL_WINDOW
    if _MASTER_STOCH_MAP is None or _MASTER_RIS_MASK is None:
        print("[ERROR] RIS Geometry not initialized for export.")
        return

    # Synchronize with actual context size (KV-Cache)
    actual_len = len(input_ids)
    last_idx = actual_len - 1
    
    # Ensure index does not overshoot precomputed geometry
    if last_idx >= _MASTER_STOCH_MAP.shape[0]:
        last_idx = _MASTER_STOCH_MAP.shape[0] - 1

    # Collect ensemble neighbors for the last actual token
    target_nodes = torch.unique(_MASTER_STOCH_MAP[last_idx]).tolist()
    if last_idx not in target_nodes:
        target_nodes.append(last_idx)
    
    # Filter nodes outside the current context (Ghost token cleanup)
    target_nodes = [idx for idx in target_nodes if idx < actual_len]
    target_nodes.sort()
    
    print(f"[RIS] Exporting Forensic Ensemble Graph ({len(target_nodes)} nodes)...")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("graph RIS_Ensemble_Brain {\n")
        # 1. Exportar labels apenas para nós válidos no contexto
        for idx in target_nodes:
            token_text = tokenizer.decode([input_ids[idx]]).replace('"', "'").replace("\n", "\\n")
            f.write(f'"{idx}" [label="[{idx}] {token_text}"];\n')
            
        edge_count = 0
        # 2. STAR Logic: Export only connections between PRESENT (last_idx) and PAST.
        # This creates an extremely lightweight recall map (300KB to 1MB).
        for v in target_nodes:
            if v == last_idx: continue
            
            # Long Range Filter: Excludes Local Window
            if abs(last_idx - v) > _RIS_LOCAL_WINDOW:
                # Check for direct stochastic connection in the ensemble
                # (last_idx attending to v OR v attending to last_idx)
                if last_idx < _MASTER_STOCH_MAP.shape[0] and v in _MASTER_STOCH_MAP[last_idx]:
                    f.write(f'"{last_idx}" -- "{v}" [WEIGHT = 1.0];\n')
                    edge_count += 1
                elif v < _MASTER_STOCH_MAP.shape[0] and last_idx in _MASTER_STOCH_MAP[v]:
                    f.write(f'"{last_idx}" -- "{v}" [WEIGHT = 1.0];\n')
                    edge_count += 1
        f.write("}\n")
    print(f"[RIS] Star Graph exported ({edge_count} edges): {filename}")

# --- CROSS-LAYER COORDINATION CACHE (Turbo 1s/token) ---
_STEP_UNION_CACHE = {}
# Generation Stochastic Anchor: Freezes heavy union of 80 seeds (context) on 1st token.
# Saves ~511 massive recalculations per response.
_GENERATION_STOCH_ANCHOR = None # (stoch_union_only)

def clear_coordination_cache():
    """Clears the RIS coordination cache to avoid pollution between conversation turns."""
    global _STEP_UNION_CACHE, _GENERATION_STOCH_ANCHOR
    _STEP_UNION_CACHE.clear()
    _GENERATION_STOCH_ANCHOR = None  # Releases the lock for the next question
    # print("[RIS] Cache de coordenação reiniciado.")

def ris_core_attention_logic(module, query, key, value, attention_mask, scaling):
    """Core sparse RIS attention logic (Optimized for CPU + Layer Caching)."""
    global _MASTER_RIS_MASK, _ACTIVE_INDEX_MAP, _ACTIVE_VALID_MASK, _STEP_UNION_CACHE, _GENERATION_STOCH_ANCHOR
    
    original_dtype = query.dtype
    bsz, _, kv_seq_len, head_dim = key.size()
    _, num_heads, q_len, _ = query.size()
    target_dtype = query.dtype if query.dtype == torch.float32 else torch.float32

    if q_len == 1:
        # [GENERATION MODE: TURBO LATENCY V9.2 - HYBRID ANCHOR]
        row_idx = kv_seq_len - 1
        if row_idx >= _ACTIVE_INDEX_MAP.shape[0]: 
            row_idx = _ACTIVE_INDEX_MAP.shape[0] - 1
        
        # 1. GENERATION HYBRID ANCHOR (Static Stochastic + Dynamic Local)
        # Heavy cost (80 unified seeds) calculated only on the first word.
        # Local window always calculated to include newly generated tokens.
        if _GENERATION_STOCH_ANCHOR is not None:
            stoch_union = _GENERATION_STOCH_ANCHOR
        else:
            # --- Computation of Stochastic Context Union (Anchor) ---
            stoch_indices = _ACTIVE_INDEX_MAP[row_idx] 
            stoch_valid   = _ACTIVE_VALID_MASK[row_idx]
            
            # Take only valid indices from the stochastic mask
            stoch_union = torch.unique(stoch_indices[stoch_valid])
            _GENERATION_STOCH_ANCHOR = stoch_union
            
        # --- Dynamic Local Window (Crucial for model self-visibility) ---
        local_start = max(0, row_idx - _RIS_LOCAL_WINDOW)
        # Local range now includes newly generated conversation/response tokens
        local_range = torch.arange(local_start, kv_seq_len, device=query.device, dtype=torch.int32)
        
        # Final Fusion: Stochastic Context + Short Term Dynamic
        union_indices = torch.unique(torch.cat([stoch_union, local_range]))
        
        union_len = union_indices.shape[0]
        union_indices_clamped = union_indices.clamp(min=0, max=kv_seq_len - 1).to(torch.long)
            
        # 2. Optimized Search in RAM (torch.index_select is faster on CPU)
        k_union = torch.index_select(key, 2, union_indices_clamped).to(target_dtype)
        v_union = torch.index_select(value, 2, union_indices_clamped).to(target_dtype)
        
        # 3. GQA Repeat (Vectorization)
        rep = module.num_key_value_groups
        if rep > 1:
            k_union = k_union.repeat_interleave(rep, dim=1)
            v_union = v_union.repeat_interleave(rep, dim=1)
            
        # 4. Unified Attention (Pre-Softmax Pooling / Unified Union Focus)
        scores_union = torch.matmul(query.to(target_dtype), k_union.transpose(-1, -2)) * scaling 
        
        # LLaMA requires weight sum to be 1.0 at the end.
        # Single softmax ensures seed frequencies don't matter anymore.
        # All selected tokens compete equally without dilution.
        probs_union = torch.nn.functional.softmax(scores_union, dim=-1, dtype=torch.float32).to(target_dtype)
        
        # 5. Direct Output
        attn_output = torch.matmul(probs_union, v_union)
    else:
        # --- PREFILL MODE ---
        # In prefill, KV repeat cost is negligible compared to global matmul
        if hasattr(modeling_qwen2, "repeat_kv"):
            key_states = modeling_qwen2.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_qwen2.repeat_kv(value, module.num_key_value_groups)
        else:
            key_states = modeling_llama.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_llama.repeat_kv(value, module.num_key_value_groups)
            
        query = query.to(torch.float32)
        key_states = key_states.to(torch.float32)
        value_states = value_states.to(torch.float32)
        
        mask_sliced = _MASTER_RIS_MASK[kv_seq_len - q_len : kv_seq_len, :kv_seq_len]
        attn_output = F.scaled_dot_product_attention(
            query, key_states, value_states,
            attn_mask=mask_sliced.unsqueeze(0).unsqueeze(0),
            is_causal=False
        )

    return attn_output.transpose(1, 2).contiguous().to(original_dtype), None

def ris_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values = None,
    **kwargs,
):
    """Robust replacement method for Qwen2Attention.forward."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # Call RIS Core
    attn_output, attn_weights = ris_core_attention_logic(
        self, query_states, key_states, value_states, attention_mask, self.scaling
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def replace_attention_with_ris(model, seq_len=65536, density=0.05, local_window=1024, global_window=256, seed=42, n_seeds=1, needs_graph=False, b_max=2048, ris_mode='stochastic', bypass_generation_map=False):
    global _RIS_CURRENT_SEED, _RIS_LOCAL_WINDOW, _RIS_GLOBAL_WINDOW, _b_max_cache, _ris_mode_cache
    _RIS_CURRENT_SEED = seed
    _RIS_LOCAL_WINDOW = local_window
    _RIS_GLOBAL_WINDOW = global_window
    _b_max_cache = b_max
    _ris_mode_cache = ris_mode
    
    _precompute_ris_structures(
        seq_len=seq_len, 
        n_seeds=n_seeds, 
        density=density, 
        w_local=local_window, 
        w_global=global_window, 
        ris_seed=seed, 
        device=next(model.parameters()).device,
        needs_graph=needs_graph,
        b_max=b_max,
        ris_mode=ris_mode,
        bypass_generation_map=bypass_generation_map
    )
    
    # Surgical Injection via Instance Method Replacement
    if hasattr(modeling_qwen2, "Qwen2Attention"):
        modeling_qwen2.Qwen2Attention.forward = ris_qwen2_attention_forward
    
    # Force Eager mode
    model.config._attn_implementation = "eager"
    return model
