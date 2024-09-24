import torch
import numpy as np

import utils
from Servers import *

if __name__ == '__main__':
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    server = BaseServer(args)
    server.run()