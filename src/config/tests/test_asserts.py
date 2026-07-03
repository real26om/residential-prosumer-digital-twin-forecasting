from config_loader import load_frozen_config
from asserts import enforce_frozen_runtime_asserts

cfg = load_frozen_config("config.yml")
enforce_frozen_runtime_asserts(cfg)
print("Frozen runtime asserts: PASS")