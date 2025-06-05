from scipy import stats
from sklearn.model_selection import ParameterGrid, ParameterSampler

enq = {"radius": [1, 2]}
enq_params = list(ParameterGrid(enq))

cdq = {"merge_strategy": ["frequency_based", "density_based"]}
cdq_params = list(ParameterGrid(cdq))

wvrn = {"max_iter": [1, 2, 3, 4, 5]}
wvrn_params = list(ParameterGrid(wvrn))

gesn = {
    "hidden_features": [512, 1024, 2048, 4096],
    "max_iterations": [30],
    "recurrent_scale": stats.loguniform(1.0, 25.0),
    "input_scale": stats.loguniform(0.1, 1.0),
    "C": stats.loguniform(1e-2, 1e3),
}
gesn_params = list(ParameterSampler(gesn, n_iter=100, random_state=42))


lr = {"C": stats.loguniform(1e-2, 1e3)}
lr_params = list(ParameterSampler(lr, n_iter=100, random_state=42))
