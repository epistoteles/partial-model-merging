from src.evaluate import evaluate_single_model
from src.utils import get_all_model_names

for model in get_all_model_names():
    try:
        evaluate_single_model(model)
    except BaseException:
        pass
