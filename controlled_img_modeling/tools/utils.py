import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def restore_weights(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location = "cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    print(f"> Restored weights from {ckpt_path}")
