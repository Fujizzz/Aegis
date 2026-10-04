"""Detached diagnostics shared by the trainer and independent inference."""
import torch
from utils.freeze_pheme import DIMS


def dimension_outputs(agents, logits, states, comm, team_layer, *, team_enabled=True):
    pre=torch.cat([torch.sigmoid(z.to(device=states.device,dtype=states.dtype)) for z in logits],dim=1)
    post=torch.cat([torch.sigmoid(a.head_on_embed(states[:,i,:].to(next(a.classifier.parameters()).device))
                    .to(device=states.device,dtype=states.dtype)) for i,a in enumerate(agents)],dim=1)
    team=torch.sigmoid(team_layer(post)).flatten().detach().cpu().tolist() if team_enabled and team_layer is not None else None
    gates=getattr(comm,'_latest_gates',None)
    gates=gates.detach().reshape(len(states),-1).mean(dim=1).cpu().tolist() if gates is not None else None
    pre,post=pre.detach().cpu().tolist(),post.detach().cpu().tolist()
    result=[]
    for index,(before,after) in enumerate(zip(pre,post)):
        row={f'{stage}_{dim}':values[i] for stage,values in (('pre',before),('post',after)) for i,dim in enumerate(DIMS)}
        row.update(team_p_positive=team[index] if team is not None else '',
                   gate_mean=gates[index] if gates is not None else '')
        result.append(row)
    return result


def parameters(modules):
    """Unique registered tensor elements (4-bit storage is not logical model size)."""
    seen=set();total=trainable=0
    for module in modules:
        if module is None:continue
        for p in module.parameters():
            if id(p) in seen:continue
            seen.add(id(p));total+=p.numel()
            if p.requires_grad:trainable+=p.numel()
    return dict(registered_parameter_elements=total,trainable_parameter_elements=trainable,
                counting_rule='Unique registered tensor numel; quantized storage is not logical dense parameter count.')


def cuda_devices(modules):
    return sorted({p.device.index for m in modules if m is not None for p in m.parameters() if p.device.type=='cuda'})


def synchronize(devices):
    for device in devices:torch.cuda.synchronize(device)


def reset_memory(devices):
    synchronize(devices)
    for device in devices:torch.cuda.reset_peak_memory_stats(device)


def memory(devices):
    synchronize(devices)
    return {str(d):dict(name=torch.cuda.get_device_name(d),peak_allocated_bytes=torch.cuda.max_memory_allocated(d),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(d)) for d in devices}


def gate_statistics(comm):
    gates=getattr(comm,'_latest_gates',None)
    if gates is None:return None
    values=gates.detach().float().flatten().cpu()
    return dict(n=values.numel(),mean=values.mean().item(),std=values.std(unbiased=False).item(),
        minimum=values.min().item(),maximum=values.max().item(),
        histogram_0_to_1=torch.histc(values,bins=10,min=0,max=1).long().cpu().tolist())
