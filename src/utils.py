import torch


def set_device(usage=50):
    device_ids = get_available_cuda(usage)
    if device_ids:
        return torch.device(f"cuda:{device_ids[0]}")
    else:
        return torch.device("cpu")

def get_available_cuda(usage=50):
    if not torch.cuda.is_available():
        return
    device_ids = []
    for i in range(torch.cuda.device_count()):
        if torch.cuda.utilization(i) < usage:
            device_ids.append(i)
    return device_ids