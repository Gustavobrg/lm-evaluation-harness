import pandas as pd

pd.DataFrame({
    "loader": ["sae_lens"],
    "action": ["add"],
    "sparse_model": ["andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts"],
    "hookpoint": ["layers.19"],
    "sae_id": ["blocks.19.hook_resid_post"],
    "feature_index": [46379],
    "steering_coefficient": [2.0],
}).to_csv("steer_config.csv", index=False)