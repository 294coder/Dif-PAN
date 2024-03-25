import torch
from warnings import warn
from collections import OrderedDict
from torch_ema import ExponentialMovingAverage


def module_load(path, model, device, ddp_rank=None, strict=True, spec_key='ema_model.shadow_params'):
    model = model.to(device if ddp_rank is None else ddp_rank)
    place = device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank}
    params = torch.load(path, map_location=place)
    
    # parse key
    parsed_keys = spec_key.split('.')
    for k in parsed_keys:
        params = params[k]

    params_load = params
    # may be tedious but useful and safe to avoid 'module.' prefix caused error
    if not strict:
        warn('model load strict is False, set it to True if you know what you are doing')
        
    def _load_fn(model, params_load, strict):    
        if 'ema' not in spec_key:  # ordered dict
            model.load_state_dict(params_load, strict=strict)
        else:  # sequential list
            for s_param, (name, param) in zip(params_load, model.named_parameters()):
                if s_param.data.shape != param.data.shape:
                    if strict:
                        raise RuntimeError('ema model load failed! shape of params does not match!')
                    else:
                        warn(f'skip the shape mismatched param, param name {name}, \
                            current shape {param.data.shape} but loaded shape {s_param.data.shape}')
                param.data.copy_(s_param.data)
    try:
        _load_fn(model, params_load, strict)
    except Exception:
        # data parallel mode will save params with keys' prefix is 'module'.
        if isinstance(params_load, dict):
            odict = OrderedDict()
            for k, v in params_load.items():
                odict['module.' + k] = v
                params[spec_key] = odict
        
        if 'ema' not in spec_key:
            _load_fn(model, params_load, strict)
        else: 
            raise RuntimeError('ema model load failed! shape of params does not match!')
            
    print('load pretrain weights')
    return model


def resume_load(path,
                model,
                optim,
                lr_scheduler,
                ema_model: ExponentialMovingAverage=None,
                specific_resume_lr: float = None,
                specific_epochs: int = None,
                wd_scheduler=None,
                device='cuda:0',
                ddp_rank=None,
                ddp=False):
    # @specific_resume_lr(warning: not recommended):
    # manually specify learning rate when the lr from last break is too low to update model

    # @specific_epochs(warning: not recommended):
    # manually specify total epochs when resuming training

    model.to(device if ddp_rank is None else ddp_rank)
    # assume saved params always on cuda:0
    params = torch.load(path, map_location=device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank})

    # NOTE: ddp mode will save params with keys' prefix is 'module'.
    #  now I remove the prefix for just one card circumstance but it conflict with ddp mode.
    if ddp:
        odict = OrderedDict()
        for k, v in params['model'].items():
            odict['module.' + k] = v
            params['model'] = odict
    model.load_state_dict(params['model'])
    
    if ema_model is not None:
        ema_model.load_state_dict(params['ema_model'])

    # NOTE: Pytorch 1.12.0 may cause CUDA error in optimizer reloading. see more at
    # https://github.com/pytorch/pytorch/issues/80809#issuecomment-1175211598
    optim.load_state_dict(params['optim'])
    if specific_resume_lr is not None:
        optim.param_groups[0]['lr'] = specific_resume_lr
        
    lr_scheduler.load_state_dict(params['lr_scheduler'])
    
    if specific_epochs is not None:
        # FIXME: only support CosineAnnealing lr_scheduler
        lr_scheduler.T_max = specific_epochs
        
    resume_ep = params['epochs']
    print(f"last training resume! best metrics are {params['metrics']}")

    # warning: if you change total epochs in the resume run, the lr_scheduler may not update lr
    if wd_scheduler is not None:
        wd_scheduler.load_state_dict(params['wd_scheduler'])
        return model, optim, lr_scheduler, wd_scheduler, resume_ep
    else:
        return model, optim, lr_scheduler, resume_ep
