import random
import sys
from your_module import load_simple_dynaMETE, run_simulation

if __name__ == '__main__':
    frac = float(sys.argv[1])
    random.seed(123)
    param, X = load_simple_dynaMETE()
    print(f"-----------Running simulation for frac={frac}----------------")
    run_simulation(X, param, frac, n_iter=20, t_max=0.05, obs_interval=0.01, start_from_prev=True)