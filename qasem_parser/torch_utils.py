import torch


def get_device(**kwargs) -> torch.cuda.Device:
    device_id = kwargs.pop("device", None)
    if device_id is None:
        device_id = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    return device
