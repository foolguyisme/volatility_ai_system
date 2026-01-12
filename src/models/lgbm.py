from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np


@dataclass
class LGBMRegressorWrapper:
    params: Dict[str, Any]
    model: Optional[lgb.LGBMRegressor] = None

    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)
