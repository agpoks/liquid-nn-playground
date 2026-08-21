"""Maps model/dataset names (as used in benchmarks/configs/*.yaml) to constructors.

Every model factory has the same signature -- (input_size, output_size, **hp) ->
nn.Module -- even though the underlying models take different hyperparameter
names (hidden_size vs. state_size, NCP's layer sizes, ...), so run_all.py can
stay generic. Every dataset entry returns (loader_fn, task_type, needs_unsqueeze).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]


def _load_model_module(name: str):
    """Every model lives in models/<name>/model.py -- all literally named
    'model.py', so a plain sys.path insert would have them clobber each other
    in sys.modules. Load each under a unique module name instead."""
    path = ROOT / "models" / name / "model.py"
    spec = importlib.util.spec_from_file_location(f"liquid_nn_playground_{name}_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RNNModel = _load_model_module("rnn").RNNModel
CTRNNModel = _load_model_module("ctrnn").CTRNNModel
CTGRUModel = _load_model_module("ctgru").CTGRUModel
NeuralODEModel = _load_model_module("node").NeuralODEModel
LiquidLSTMModel = _load_model_module("liquid_lstm").LiquidLSTMModel
LTCModel = _load_model_module("ltc").LTCModel
CfCModel = _load_model_module("cfc").CfCModel
NCPModel = _load_model_module("ncp").NCPModel
LiquidS4Model = _load_model_module("liquid_s4").LiquidS4Model
LrcSSMModel = _load_model_module("lrcssm").LrcSSMModel

from liquid_playground.data import (  # noqa: E402
    load_ett,
    load_ozone,
    load_person_activity,
    load_room_occupancy,
    load_sequential_mnist,
)


def _rnn_factory(input_size, output_size, hidden_size=32, **_):
    return RNNModel(input_size, hidden_size, output_size)


def _ctrnn_factory(input_size, output_size, hidden_size=32, **_):
    return CTRNNModel(input_size, hidden_size, output_size)


def _ctgru_factory(input_size, output_size, hidden_size=32, **_):
    return CTGRUModel(input_size, hidden_size, output_size)


def _node_factory(input_size, output_size, hidden_size=32, **_):
    return NeuralODEModel(input_size, hidden_size, output_size)


def _liquid_lstm_factory(input_size, output_size, hidden_size=32, **_):
    return LiquidLSTMModel(input_size, hidden_size, output_size)


def _ltc_factory(input_size, output_size, hidden_size=32, **_):
    return LTCModel(input_size, hidden_size, output_size)


def _cfc_factory(input_size, output_size, hidden_size=32, **_):
    return CfCModel(input_size, hidden_size, output_size)


def _ncp_factory(input_size, output_size, **_):
    return NCPModel(input_size, output_size)


def _liquid_s4_factory(input_size, output_size, state_size=32, n_layers=2, **_):
    return LiquidS4Model(input_size, state_size, output_size, n_layers=n_layers)


def _lrcssm_factory(input_size, output_size, state_size=32, n_layers=2, **_):
    return LrcSSMModel(input_size, state_size, output_size, n_layers=n_layers)


MODEL_REGISTRY: dict[str, Callable] = {
    "rnn": _rnn_factory,
    "ctrnn": _ctrnn_factory,
    "ctgru": _ctgru_factory,
    "node": _node_factory,
    "liquid_lstm": _liquid_lstm_factory,
    "ltc": _ltc_factory,
    "cfc": _cfc_factory,
    "ncp": _ncp_factory,
    "liquid_s4": _liquid_s4_factory,
    "lrcssm": _lrcssm_factory,
}


def _ozone_loader(**kw):
    tr_x, tr_y, te_x, te_y = load_ozone(**kw)
    return tr_x.unsqueeze(-1), tr_y, te_x.unsqueeze(-1), te_y


def _mnist_loader(**kw):
    return load_sequential_mnist(**kw)


def _ett_loader(**kw):
    tr_x, tr_y, te_x, te_y = load_ett(**kw)
    return tr_x, tr_y.reshape(tr_y.shape[0], -1), te_x, te_y.reshape(te_y.shape[0], -1)


# name -> (loader, task_type, output_size or None to infer from labels)
DATASET_REGISTRY: dict[str, tuple[Callable, str, int | None]] = {
    "ozone": (_ozone_loader, "binary", 1),
    "room_occupancy": (load_room_occupancy, "binary", 1),
    "person_activity": (load_person_activity, "multiclass", None),
    "sequential_mnist": (_mnist_loader, "multiclass", 10),
    "ett": (_ett_loader, "regression", None),
}
