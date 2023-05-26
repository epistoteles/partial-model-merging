import matplotlib.pyplot as plt
import seaborn as sns

from src.models.VGG import VGG
from src.utils.data_utils import load_model, normalize


MODEL_TO_PLOT = "VGG11-1x-a"
model = VGG(11)

model = load_model(model, MODEL_TO_PLOT)
sd = model.state_dict()

for i, key in enumerate(sd.keys()):
    if "features" in key and "weight" in key:
        pass
