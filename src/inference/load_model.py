import os
import lightgbm as lgb


def load_lgbm_model(model_path: str) -> lgb.Booster:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return lgb.Booster(model_file=model_path)
