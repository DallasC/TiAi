# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from train_models.ddpg import DDPG

__all__ = ["DDPG"]

