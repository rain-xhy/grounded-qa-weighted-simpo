def chat_inputs(tok, prompt):
    if not hasattr(tok, 'build_chat_input'):
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                      add_generation_prompt=True, return_tensors='pt', return_dict=True)
    inputs = tok.build_chat_input(prompt, history=[], role='user')
    inputs.pop('token_type_ids', None)
    return inputs

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
