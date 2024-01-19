import torch

import os, sys


script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.9/dist-packages/lmi_dist")

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
from djl_python.tests.rolling_batch_test_scripts.generator import Generator, print_rank0


