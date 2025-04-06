# ========================== IDS Models Runner =========================
#
#                   Author:  Sergio Arroni Del Riego
#
# ======================================================================

# ==================> Imports
from intrusion_detection_systems.models import k_neig_model, dec_tree_model, r_forest_model, log_reg_model, mlp_model, svc_model, nb_model, cnn_model
from intrusion_detection_systems.metrics import CM, SMT, SMLM
from shared.utils import save_model
from typing import Any
import numpy as np

# ==================> Enumerations
models_types_default = {
    "KNN": k_neig_model,
    "DT": dec_tree_model,
    "RF": r_forest_model,
    "LR": log_reg_model,
    "MLP": mlp_model,
    "SVC": svc_model,
    "NB": nb_model,
    "CNN": cnn_model
}

metrics_types = {"CM": CM, "SMT": SMT, "SMLM": SMLM}

# ==================> Functions


def train_ids_model(x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, models_type: list,
                    save: bool, seed: int) -> Any:
    models = []
    for model in models_type:
        # Xử lý đặc biệt cho CNN
        if model == "CNN":
            # Reshape dữ liệu cho CNN (thêm chiều thứ 3)
            x_train_reshaped = np.expand_dims(x_train, axis=-1)
            x_test_reshaped = np.expand_dims(x_test, axis=-1)
            
            model_t = models_types_default[model](
                x_train=x_train_reshaped, 
                y_train=y_train, 
                x_test=x_test_reshaped,
                y_test=y_test, 
                dataset=dataset, 
                seed=seed)
        else:
            model_t = models_types_default[model](
                x_train=x_train, 
                y_train=y_train, 
                x_test=x_test,
                y_test=y_test, 
                dataset=dataset, 
                seed=seed)
            
        if save:
            save_model(model_t, dataset + "_" + model)
            print(f"Model {model} saved")
        models.append(model_t)
    return models


def show_model_metrics(model: Any, metric_type: str) -> None:
    """show_model_metrics

    This function shows the metrics of the given model in the console

    Parameters:
        model: Model
    Output:
        None
    """
    metrics_types[metric_type](model).operation()
    '''
    CM(model).operation()
    SMT(model).operation()
    SMLM(model).operation()
    '''