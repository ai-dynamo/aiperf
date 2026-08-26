// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! In-process Optuna operations for `--search-style bo|optuna`.
//!
//! [`crate::bayes`] owns orchestration and convergence. This module owns the
//! study, pending trial, sampler, and constraint callback. Positive constraint
//! values violate the SLA; non-positive values are feasible.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};

/// Embedded Optuna helper module. `sla_specs` is a list of
/// `(attr_key, op, threshold)`; `constraints_func` reads each observation off
/// `trial.user_attrs[attr_key]` and returns the signed violation.
const PYOPT_SRC: &str = r#"
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

_UNMEASURABLE_VIOLATION = 1.0e6


def _signed_violation(observed, op, threshold):
    if observed is None:
        return _UNMEASURABLE_VIOLATION
    if op in ("lt", "le"):
        return observed - threshold
    return threshold - observed


def make_study(direction, sampler_name, n_startup, seed, sla_specs):
    specs = [(k, op, float(thr)) for (k, op, thr) in sla_specs]

    def constraints_func(trial):
        return [_signed_violation(trial.user_attrs.get(k), op, thr) for (k, op, thr) in specs]

    cf = constraints_func if specs else None
    if sampler_name == "tpe":
        from optuna.samplers import TPESampler

        sampler = TPESampler(n_startup_trials=n_startup, seed=seed, constraints_func=cf)
    elif sampler_name == "gp":
        from optuna.samplers import GPSampler

        sampler = GPSampler(n_startup_trials=n_startup, seed=seed, constraints_func=cf)
    elif sampler_name in ("botorch", "bo"):
        # `bo` selects qLogNEI; `botorch` uses BoTorch's constrained default.
        from optuna_integration import BoTorchSampler

        candidates_func = None
        if sampler_name == "bo":
            from aiperf.orchestrator.search_planner._optuna_helpers import (
                build_qlognei_candidates_func,
            )

            candidates_func = build_qlognei_candidates_func()
        sampler = BoTorchSampler(
            n_startup_trials=n_startup,
            seed=seed,
            constraints_func=cf,
            candidates_func=candidates_func,
        )
    else:
        raise ValueError(f"unsupported sampler {sampler_name!r}")
    return optuna.create_study(directions=[direction], sampler=sampler)


def ask_int(study, path, lo, hi, log):
    trial = study.ask()
    v = trial.suggest_int(path, int(lo), int(hi), log=bool(log))
    return trial, int(v)


def tell(study, trial, objective, attr_items):
    for k, val in attr_items:
        trial.set_user_attr(k, val)
    study.tell(trial, [float(objective)])
"#;

/// A live optuna study plus its pending trial, held across ask/tell.
pub struct OptunaStudy {
    module: Py<PyModule>,
    study: Py<PyAny>,
    pending: Option<Py<PyAny>>,
}

impl OptunaStudy {
    /// Create a study, seeded only when `seed` is present. `direction` is
    /// `"maximize"`/`"minimize"`; `sampler_name` is `"tpe"`/`"gp"`/`"botorch"`/`"bo"`;
    /// `sla_specs` is `(attr_key, op, threshold)` per SLA filter (empty =
    /// unconstrained).
    pub fn new(
        direction: &str,
        sampler_name: &str,
        n_startup: i64,
        seed: Option<u64>,
        sla_specs: &[(String, String, f64)],
    ) -> anyhow::Result<Self> {
        Python::with_gil(|py| -> PyResult<Self> {
            let module = PyModule::from_code(
                py,
                std::ffi::CString::new(PYOPT_SRC)?.as_c_str(),
                std::ffi::CString::new("pyopt.py")?.as_c_str(),
                std::ffi::CString::new("aiperf_pyopt")?.as_c_str(),
            )?;
            let specs = PyList::empty(py);
            for (k, op, thr) in sla_specs {
                specs.append(PyTuple::new(
                    py,
                    [
                        k.into_pyobject(py)?.into_any(),
                        op.into_pyobject(py)?.into_any(),
                        thr.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let study = module.getattr("make_study")?.call1((
                direction,
                sampler_name,
                n_startup,
                seed,
                specs,
            ))?;
            Ok(Self {
                module: module.unbind(),
                study: study.unbind(),
                pending: None,
            })
        })
        .map_err(|e: PyErr| anyhow::anyhow!("optuna seam failed (is optuna importable?): {e}"))
    }

    /// Ask the study for the next integer suggestion on `path` in `[lo, hi]`
    /// (log scale when `log`). Latches the pending trial for the next [`Self::tell`].
    pub fn ask_int(&mut self, path: &str, lo: i64, hi: i64, log: bool) -> anyhow::Result<i64> {
        Python::with_gil(|py| -> PyResult<i64> {
            let out = self.module.bind(py).getattr("ask_int")?.call1((
                self.study.bind(py),
                path,
                lo,
                hi,
                log,
            ))?;
            let (trial, value): (Bound<'_, PyAny>, i64) = out.extract()?;
            self.pending = Some(trial.unbind());
            Ok(value)
        })
        .map_err(|e: PyErr| anyhow::anyhow!("optuna ask failed: {e}"))
    }

    /// Tell the study the `objective` for the pending trial, first writing each
    /// `(attr_key, value)` SLA observation onto `trial.user_attrs` (None = missing).
    pub fn tell(&mut self, objective: f64, attrs: &[(String, Option<f64>)]) -> anyhow::Result<()> {
        let trial = self
            .pending
            .take()
            .ok_or_else(|| anyhow::anyhow!("tell without ask"))?;
        Python::with_gil(|py| -> PyResult<()> {
            let items = PyList::empty(py);
            for (k, v) in attrs {
                let val = match v {
                    Some(x) => x.into_pyobject(py)?.into_any(),
                    None => py.None().into_bound(py),
                };
                items.append(PyTuple::new(py, [k.into_pyobject(py)?.into_any(), val])?)?;
            }
            self.module.bind(py).getattr("tell")?.call1((
                self.study.bind(py),
                trial.bind(py),
                objective,
                items,
            ))?;
            Ok(())
        })
        .map_err(|e: PyErr| anyhow::anyhow!("optuna tell failed: {e}"))
    }
}
