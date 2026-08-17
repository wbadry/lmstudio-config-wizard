def _parse_gpu_memory_mb(value):
    """Normalize numeric or legacy '<number> MB' GPU memory values."""
    if isinstance(value, str):
        value = value.strip().split(maxsplit=1)[0]

    try:
        return max(0, round(float(value)))
    except (ValueError, TypeError):
        return 0


def recommend_settings(hardware_info: dict, model_profile: dict) -> dict:
    """
    Generate recommended LM Studio configuration values based on hardware and user needs.
    """
    # Fix: Use correct keys from hardware_info and ensure int conversion for cores
    # 'ram_gb' and 'logical_cores' are the keys returned by hardware_utils.py
    ram_gb = hardware_info.get("ram_gb", 8.0) # Ensure float for comparison
    cpu_logical_cores = int(hardware_info.get("logical_cores", 4)) # Ensure int for calculations

    gpu_name = hardware_info.get("gpu", "None")
    gpu_memory_mb = _parse_gpu_memory_mb(hardware_info.get("gpu_memory_mb", 0))

    # Fix: Use 'goal' from model_profile, as 'usage_needs' is not returned by model_profile.py
    user_goal = model_profile.get("goal", "Balanced/general purpose")

    # --- Context Length ---
    # Default context length based on general RAM, will be primarily overridden by model size below
    context_length = 2048 
    if ram_gb >= 32.0:
        context_length = 8192
    elif ram_gb >= 16.0:
        context_length = 4096

    # Override context_length based on model_size mapping for a more precise fit
    model_size = model_profile.get("model_size", "Other")
    size_context_recommendations = {
        "Under 2 GB": 1024,
        "2-4 GB": 2048,
        "4-8 GB": 4096,
        "8-13 GB": 8192,
        "More than 13 GB": 16384, # This can be very high and might still be challenging for some setups.
                                   # User might need to adjust downwards if out of memory issues persist.
        "Other": 2048,
    }
    context_length = size_context_recommendations.get(model_size, context_length)


    # --- GPU Offload ---
    gpu_offload = 0 # Default to no offload

    # Determine if a capable GPU is present (e.g., Apple M series, or a discrete GPU with significant VRAM)
    is_capable_gpu = False
    if "Apple M" in gpu_name:
        is_capable_gpu = True # Assume Apple M-series chips are capable for offloading
    elif gpu_memory_mb >= 4096: # For other GPUs, check for at least 4GB VRAM
        is_capable_gpu = True

    if is_capable_gpu:
        # For capable GPUs, aim to offload as many layers as possible.
        # In LM Studio, a high number like 999 or -1 often means "all possible layers".
        # We'll use 999 here as it's a common practice.
        gpu_offload = 999
        
        # Although quantization_recommendations exist, for powerful GPUs, we want to maximize offload.
        # The specific quantization types (Q4_K_M, Q4_K_S, Q8_0) generally benefit from more offload.
        # If logic for specific layers per quantization is desired, it would need more granular data.
    else:
        # If no capable GPU detected or insufficient memory, stick to no offload
        gpu_offload = 0

    # --- CPU Threads ---
    # Use logical cores for CPU threads, capped at a reasonable number to leave resources for system/GPU.
    # A value of 8 or 12 is often a good balance for systems with many cores.
    cpu_threads = min(cpu_logical_cores, 8) 


    # --- Evaluation Batch Size ---
    # Default batch size, will be primarily overridden by model size below
    batch_size = 1 # Start very low for safety for large models to prevent OOM errors

    # Override batch_size based on model_size mapping
    size_batch_recommendations = {
        "Under 2 GB": 16,
        "2-4 GB": 8,
        "4-8 GB": 4,
        "8-13 GB": 2,
        "More than 13 GB": 1, # Crucial: Set to 1 for very large models to prevent Out Of Memory
        "Other": 1,
    }
    batch_size = size_batch_recommendations.get(model_size, batch_size)


    # --- Creativity Parameters (Generation Settings) ---
    # Dynamically set temperature, top_p, and top_k based on user's goal
    if "Creativity" in user_goal:
        temp = 1.0
        top_p = 0.95
        top_k = 100
    elif "Accuracy" in user_goal:
        temp = 0.3
        top_p = 0.85
        top_k = 50
    elif "Natural Dialogue" in user_goal:
        temp = 0.7
        top_p = 0.9
        top_k = 40
    elif "Factual Recall" in user_goal:
        temp = 0.4
        top_p = 0.8
        top_k = 30
    else:  # Balanced/general purpose, as the default if no specific goal is matched
        temp = 0.7
        top_p = 0.9
        top_k = 60

    # --- Fixed Parameters ---
    repeat_penalty = 1.1
    max_tokens = 1024 # Standard max tokens, user can adjust based on desired output length

    # --- Memory Management Parameters ---
    # Fix: Correctly use the 'ram_gb' variable (now lowercased) for comparison
    keep_model_in_memory = ram_gb >= 24.0 # Keep model in memory if sufficient RAM is available
    
    # Flash attention typically requires specific GPU architectures and sufficient VRAM.
    # On Apple Silicon, unified memory means VRAM detection can be tricky, so we assume capability.
    flash_attention = False
    if "Apple M" in gpu_name: # Apple M-series chips generally support optimized attention mechanisms
        flash_attention = True
    elif gpu_memory_mb >= 8192: # For other GPUs, check explicit VRAM threshold
        flash_attention = True

    try_mmap = True # Generally beneficial for GGUF models, allows faster loading

    # --- Assemble final configuration dictionary ---
    config = {
        "context_length": context_length,
        "gpu_offload": gpu_offload,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "max_tokens": max_tokens,
        "keep_model_in_memory": keep_model_in_memory,
        "flash_attention": flash_attention,
        "try_mmap": try_mmap
    }

    return config
