import torch
from functools import partial

from utils._metric_legacy import analysis_accu
from utils.metric import (
    psnr_batch_tensor_metric,
    ssim_batch_tensor_metric,
    normalize_to_01,
    AnalysisPanAcc,
    NonAnalysis,
)
from utils.metric_FLIR import AnalysisFLIRAcc
from utils.print_helper import log, warning, error
from utils.log_utils import (
    get_logger,
    WandbLogger,
    TensorboardLogger,
    NoneLogger,
    ep_loss_dict2str,
)
from utils.log_utils import TrainStatusLogger as TrainProcessTracker
from utils.misc import (
    dict_to_str,
    to_device,
    prefixed_dict_key,
    is_main_process,
    print_args,
    to_numpy,
    Identity,
    merge_args_namespace,
    generate_id,
    set_all_seed,
    exists,
    default,
    yaml_load,
    json_load,
    config_py_load,
    h5py_to_dict,
    print_network_params_macs_fvcore,
)
from utils.misc import recursive_search_dict2namespace as convert_config_dict
from utils.load_params import module_load, resume_load
from utils.optim_utils import (
    cosine_scheduler,
    get_scheduler,
    get_optimizer,
    LinearWarmupScheduler,
)
from utils.network_utils import (
    variance_scaling_initializer,
    clip_norm,
    loss_with_l2_regularization,
    step_loss_backward,
    hook_model,
)
from utils.visualize import viz_batch, res_image, get_spectral_image_ready
from utils.inference_helper_func import ref_for_loop, unref_for_loop
from utils.loss_utils import get_loss, accum_loss_dict, ave_ep_loss, ave_multi_rank_dict
from utils.save_checker import BestMetricSaveChecker


config_load = yaml_load

if __name__ == "__main__":
    import torch

    a = torch.randn(3, 3, 256, 256)
    b = torch.randn(3, 3, 256, 256)
    analysis = AnalysisPanAcc()
    d = analysis(a, b)
    print(analysis.print_str())

    d = analysis(a, b)
    print(analysis.print_str())
