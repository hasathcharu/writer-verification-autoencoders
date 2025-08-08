from modules.utils.eval_utils import (
    evaluate_autoencoders_sample_wise,
    print_results,
    save_sample_results
)

import os
from modules.utils import constants as c
import multiprocessing
import numpy as np

if __name__ == "__main__":

    N_CORES = multiprocessing.cpu_count()

    dirs = sorted(os.listdir(f"{c.FEATURES}/{c.TRAIN}"))
    writer_ids = [dir.split(".")[0] for dir in dirs]

    global_results = evaluate_autoencoders_sample_wise(writer_ids)

    print("Results:")
    print_results(global_results)
    save_sample_results(global_results, f"{c.RESULTS}/autoencoders", c.FEATURES)
