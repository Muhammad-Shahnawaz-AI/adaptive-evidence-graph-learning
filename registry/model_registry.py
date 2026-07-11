MODEL_REGISTRY = {}

def register(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]