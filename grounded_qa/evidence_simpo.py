"""Exact SimPO gradient with sequential graphs and reproducible dropout replay."""
def simpo_loss(chosen_nll,rejected_nll,beta=10.,gamma_beta_ratio=.3,sft_weight=0.):
    import torch.nn.functional as F
    return F.softplus(-beta*(rejected_nll-chosen_nll-gamma_beta_ratio))+sft_weight*chosen_nll


def rng_state():
    import torch
    return (torch.random.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    import torch
    torch.random.set_rng_state(value[0])
    if value[1] is not None:torch.cuda.set_rng_state_all(value[1])


def backward_pair(loss_fn,beta=10.,gamma_beta_ratio=.3,sft_weight=0.,accumulation=1,scaler=None):
    """loss_fn(bool) returns mean response CE; no training target truncation."""
    import torch
    if beta<=0 or accumulation<1:raise ValueError('Invalid SimPO parameters')
    chosen_rng=rng_state()
    with torch.no_grad():chosen=loss_fn(True).detach()
    rejected_rng=rng_state()
    with torch.no_grad():rejected=loss_fn(False).detach()
    finished_rng=rng_state()
    if not torch.isfinite(chosen) or not torch.isfinite(rejected):raise ValueError('Nonfinite pair loss')
    coefficient=(beta*torch.sigmoid(-beta*(rejected-chosen-gamma_beta_ratio))).detach()
    def back(value):
        (scaler.scale(value) if scaler else value).backward()
    try:
        restore_rng(chosen_rng);chosen_graph=loss_fn(True)
        if not torch.allclose(chosen_graph.detach(),chosen,rtol=1e-4,atol=1e-4):
            raise ValueError('Chosen dropout replay or loss identity differs')
        back((coefficient+sft_weight)*chosen_graph/accumulation)
        del chosen_graph
        restore_rng(rejected_rng);rejected_graph=loss_fn(False)
        if not torch.allclose(rejected_graph.detach(),rejected,rtol=1e-4,atol=1e-4):
            raise ValueError('Rejected dropout replay or loss identity differs')
        back(-coefficient*rejected_graph/accumulation)
        del rejected_graph
    finally:restore_rng(finished_rng)
    return {'loss':float(simpo_loss(chosen,rejected,beta,gamma_beta_ratio,sft_weight)),
            'chosen_nll':float(chosen),'rejected_nll':float(rejected),
            'preference_margin':float(rejected-chosen),'coefficient':float(coefficient)}
