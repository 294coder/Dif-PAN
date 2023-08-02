from utils.log_utils import AimLogger

logger = AimLogger("test_aim_v2", hparams={"net_name": "test_net"})

logger.log_text("nothing", "try again")

import numpy as np

a = np.random.randn(1, 2000)
logger.log_distribution(a, "test_dist")

# import torch

# conv = torch.nn.Conv2d(3, 3, 3)

# logger.log_network(conv)

# import matplotlib.pyplot as plt
# import numpy as np

# plt.imshow(np.random.randn(256, 256))
# fig = plt.gcf()

# logger.log_image(fig, name="figs", context={"train": "figs"})

logger.close()