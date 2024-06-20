from collections import OrderedDict
import numpy as np
from scipy import integrate
import cmath
from tqdm.auto import tqdm
import torch


def generate_interference(model_parameters, num_servers, servers_xx, servers_yy, devices_xx, devices_yy):
    interference = OrderedDict()

    def f(x):
        return (1 / x) * np.exp(-x)

    Ei, err = integrate.quad(f, 0.2, np.inf)

    rho = (6*0.25) / (2 * Ei * 0.016)

    for key in model_parameters.keys():
        # mean = torch.mean(model_parameters[key]).item()
        # var = torch.var(model_parameters[key]).item()
        for cluster_idx in range(num_servers-1):
            num_selected_clients = 10
            print(devices_xx)
            print(len(devices_xx[cluster_idx]))
            selected_client_indices = sorted(np.random.choice(a=[i for i in range(len(devices_xx[cluster_idx]))], size=num_selected_clients, replace=False).tolist())
            for device_idx in tqdm(selected_client_indices, leave=False):
                h1_squared = np.random.exponential(1, 1)[0]
                if h1_squared < 0.2:
                    continue
                r1 = np.sqrt((devices_xx[cluster_idx][device_idx]-servers_xx[cluster_idx]) ** 2 + (devices_yy[cluster_idx][device_idx]-servers_yy[cluster_idx]) ** 2)
                power = np.sqrt(rho) * (r1 ** 1.5) / cmath.rect(np.sqrt(h1_squared), np.random.uniform(0, 2 * cmath.pi))
                h2 = cmath.rect(np.sqrt(np.random.exponential(1, 1)[0]), np.random.uniform(0, 2 * cmath.pi))
                r2 = np.sqrt((devices_xx[cluster_idx][device_idx]) ** 2 + (devices_yy[cluster_idx][device_idx]) ** 2)
                coefficient = (power * h2) / (r2 ** 1.5)
                # if key not in interference.keys():
                #     interference[key] = coefficient.real * torch.tensor(np.random.normal(mean, var, tuple(model_parameters[key].shape)), dtype=torch.float32)
                    # param_tensor = model_parameters[key].clone()
                    # scaled_tensor = coefficient.real*param_tensor
                    # interference[key] = scaled_tensor
                    # interference[key] = coefficient.real * torch.tensor(model_parameters[key], dtype=torch.float32)
                # else:
                #     interference[key] += coefficient.real * torch.tensor(np.random.normal(mean, var, tuple(model_parameters[key].shape)), dtype=torch.float32)
                    # param_tensor = model_parameters[key].clone()
                    # scaled_tensor = coefficient.real * param_tensor
                    # interference[key] += scaled_tensor
                    # interference[key] += coefficient.real * torch.tensor(model_parameters[key], dtype=torch.float32)
                device = model_parameters[key].device
                if key not in interference.keys():
                    interference[key] = torch.tensor(coefficient.real, device=device) * model_parameters[key].to(device)
                else:
                    interference[key] += torch.tensor(coefficient.real, device=device) * model_parameters[key].to(device)
    return interference, rho

