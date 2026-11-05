"""Shared model loading, memory-efficient forwards and response log probabilities."""


def chat_inputs(tok, prompt):
    if not hasattr(tok, 'build_chat_input'):
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                      add_generation_prompt=True, return_tensors='pt', return_dict=True)
    inputs = tok.build_chat_input(prompt, history=[], role='user')
    inputs.pop('token_type_ids', None)
    return inputs


def memory_adapt(model, chunk_size=512):
    """Token-independent MLP chunks and last-token-only logits for generation.

    Reward scoring uses the decoder and unwrapped lm_head separately. Neither
    optimization truncates source tokens or changes model parameters.
    """
    import torch
    if hasattr(model.lm_head, '_case5_full_forward'):
        return
    full_head = model.lm_head.forward
    model.lm_head._case5_full_forward = full_head
    model.lm_head.forward = lambda hidden: full_head(hidden[:, -1:, :])
    if chunk_size:
        for layer in model.model.layers:
            original = layer.mlp.forward
            def chunked(hidden, original=original):
                if hidden.shape[1] <= chunk_size:
                    return original(hidden)
                return torch.cat([original(hidden[:, i:i+chunk_size])
                                  for i in range(0, hidden.shape[1], chunk_size)], dim=1)
            layer.mlp.forward = chunked


def load_model(args, training=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from longcite_modeling_llama import LlamaForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("The QLoRA runner requires a CUDA GPU")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True,
                                              trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    base = LlamaForCausalLM.from_pretrained(args.model, local_files_only=True, device_map={"": 0},
                                           torch_dtype=torch.float16, attn_implementation="sdpa",
                                           quantization_config=quant)
    memory_adapt(base, 512)
    if training:
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.adapter:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=training)
    elif training:
        cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=.05,
                         target_modules=["q_proj", "v_proj"], bias="none")
        model = get_peft_model(base, cfg)
    else:
        model = base
    model.train(training)
    return tokenizer, base, model


def target_token_loss(head, hidden, labels, chunk_size=64, checkpoint_chunks=True):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    if hidden.shape[:-1] != labels.shape or labels.numel() == 0 or chunk_size < 1:
        raise ValueError('Invalid response loss shapes or chunk size')
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    loss = hidden.new_zeros((), dtype=torch.float32)
    def compute(h, y):
        return F.cross_entropy(head(h).float(), y, reduction='sum')
    for index in range(0, labels.numel(), chunk_size):
        h, y = hidden[index:index+chunk_size], labels[index:index+chunk_size]
        value = (checkpoint(compute, h, y, use_reentrant=False)
                 if checkpoint_chunks and torch.is_grad_enabled() else compute(h, y))
        loss = loss+value
    return loss/labels.numel()


def encoded_loss(tokenizer, base, prompt, response, limit, history="", add_eos=True):
    import torch
    prefix = chat_inputs(tokenizer, prompt)["input_ids"][0].tolist()
    prefix += tokenizer.encode(history, add_special_tokens=False)
    target = tokenizer.encode(response, add_special_tokens=False)
    if not target:
        raise ValueError("Empty response")
    if add_eos:
        target.append(tokenizer.eos_token_id)
    if len(prefix) + len(target) > limit:
        raise ValueError("Input and target exceed the token budget")
    ids = torch.tensor([prefix + target], device=base.device)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            return_dict=True).last_hidden_state[:, len(prefix)-1:-1]
        return target_token_loss(base.lm_head._case5_full_forward, hidden, ids[:, len(prefix):])
