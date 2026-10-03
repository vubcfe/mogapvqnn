"""Tuned classical references and their merge into the report."""

import numpy as np

from mogapvqnn.baselines import TUNE_C, tuned_svm, tuning_grid
from mogapvqnn.classifier import accuracy, fit_linear
from mogapvqnn.data import synthetic_task
from mogapvqnn.report import merge_classical, present_methods


def _task():
    return synthetic_task(4, 80, 40, 0)


def test_grids_contain_untuned_setting():
    assert 1.0 in TUNE_C
    assert {"C": 1.0} in tuning_grid("linear")
    assert all(p["coef0"] == 0.0 for p in tuning_grid("rbf"))
    assert {p["coef0"] for p in tuning_grid("poly2")} == {0.0, 1.0}


def test_tuned_linear_not_worse_on_validation():
    t = _task()
    _, params, val = tuned_svm("linear", t.X_fit, t.y_fit, t.X_val, t.y_val)
    base = accuracy(fit_linear(t.X_fit, t.y_fit, 1.0).predict(t.X_val), t.y_val)
    assert val >= base
    assert params["C"] in TUNE_C


def test_tuned_svm_deterministic():
    t = _task()
    for kind in ("linear", "poly2", "rbf"):
        m1, p1, v1 = tuned_svm(kind, t.X_fit, t.y_fit, t.X_val, t.y_val)
        m2, p2, v2 = tuned_svm(kind, t.X_fit, t.y_fit, t.X_val, t.y_val)
        assert p1 == p2 and v1 == v2
        assert np.array_equal(m1.predict(t.X_test), m2.predict(t.X_test))


def test_merge_classical_in_memory_only():
    run = {"ensembles": {"greedy": {"test_acc": 0.7}}, "baselines": {"linear": {"test_acc": 0.8}}}
    full = {"cfg_4q": {0: run}}
    extra = {"cfg_4q": {0: {"baselines": {"rbf_cv": {"test_acc": 0.9}}}}}
    assert merge_classical(full, extra) == 1
    assert full["cfg_4q"][0]["baselines"]["rbf_cv"]["test_acc"] == 0.9
    assert present_methods(full, ["mo_ga", "linear", "poly2", "rbf_cv"]) == ["mo_ga", "linear", "rbf_cv"]
