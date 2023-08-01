import torch
from collections import OrderedDict


def module_load(path, model, device, ddp_rank, strict=True):
    model = model.to(device if ddp_rank is None else ddp_rank)
    place = device if ddp_rank is None else {'cuda:%d' % 0: 'cuda:%d' % ddp_rank}
    params = torch.load(path, map_location=place)

    # may be tedious but useful and safe to avoid 'module.' prefix caused error
    if not strict:
        print('warning: model load strict is False, ' 
              'set it to True if you know what you are doing')
    try:
        model.load_state_dict(params['model'], strict=strict)
    except Exception:
        odict = OrderedDict()
        for k, v in params['model'].items():
            odict['module.' + k] = v
            params['model'] = odict
        model.load_state_dict(params['model'], strict=strict)
    print('load pretrain weights')
    return model


def resume_load(path,
                model,
                optim,
                lr_scheduler,
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
