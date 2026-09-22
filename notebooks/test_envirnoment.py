import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost
import catboost
import torch
import transformers

print("=" * 50)
print("Environment Test")
print("=" * 50)

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("LightGBM:", lightgbm.__version__)
print("XGBoost:", xgboost.__version__)
print("CatBoost:", catboost.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 50)
print("ENVIRONMENT READY")
print("=" * 50)