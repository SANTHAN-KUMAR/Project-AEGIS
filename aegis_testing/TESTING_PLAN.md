# AEGIS 3.0 — Comprehensive Testing Strategy (v4 Final)

> **Objective**: Rigorous first-principles validation of all 5 layers + cross-layer integration.
> **Total Tests**: 136 (95 new + 41 existing)
> **Estimated Compute**: ~100-140 hours

This document outlines the definitive testing plan for AEGIS 3.0, incorporating extensive peer review and critical feedback. It defines the exact pass criteria, failure protocols, and execution order for validating the system.

---

## 🚦 Failure Protocol

If a test fails, triage according to this protocol:

| Severity | Type | Action | Examples |
|---|---|---|---|
| 🔴 | **BLOCKING** | **Stop immediately.** Downstream tests are invalid. | • `T0.3` (simglucose divergence)<br>• `T2.11` (Calibration fails → L5 void)<br>• `T2.1a` (UDE hurts performance) |
| 🟡 | **NON-BLOCKING** | **Continue execution**, fix later. | • `T2.1b`, `T3.3b`, `T4.7`, `T4.8`<br>• Distribution shift tests (`T2.12`, `T3.10`) |
| 🔵 | **PAPER-SCOPING** | **Adjust paper claims**, do not block. | • `T3.7` fails → Remove "detects individual effects"<br>• `T4.2` fails → Remove "√T regret" claim<br>• `CL.1b` fails → Remove "safe dosing" claim |

---

## 🏗️ Phase 0: Infrastructure (BLOCKING)

**Status**: ✅ **17/17 PASSED** (2026-02-15)

| Test | Objective | Pass Criterion |
|---|---|---|
| **T0.1** | Deterministic reproducibility | 3 runs produce bit-identical results |
| **T0.2** | Simulation stability | All 30 patients load + 24h sim (no NaN/Inf) |
| **T0.3** | Bergman parameter cross-check | Document divergence (JS prototype vs simglucose) |

> **Critical Finding**: JS prototype hardcoded parameters diverge significantly from `simglucose` native params. L2 must use `simglucose` values.

---

## 🧠 Layer 1: Semantic Sensorium

**Compute**: ~2h | **Files**: `aegis_core/semantic_sensorium.py`

| ID | Test Name | Runs | Pass Criterion |
|---|---|---|---|
| **T1.1** | Concept extraction (held-out) | 500 | F1 ≥ 0.80 |
| **T1.2** | Semantic entropy calibration | 200 | Spearman ρ ≥ 0.60, AUC ≥ 0.80 |
| **T1.3** | HITL trigger precision | 200 | Capture ≥ 85%, FAR ≤ 50% |
| **T1.4** | Non-circular proxy classif. | 1,000 | P ≥ 0.80, R ≥ 0.75 (held-out patterns) |
| **T1.5** | Causal bias reduction | 8,000 | Bias reduction ≥ 30% vs no-proxy |
| **T1.6** | Text noise robustness | 2,000 | F1 ≥ 0.70 with typos/negation |
| **T1.8** | **Negative Control** | 200 | F1 ≈ 0 on non-medical text |

---

## 🧬 Layer 2: Adaptive Digital Twin

**Compute**: ~50-80h (GPU) | **Files**: `aegis_core/digital_twin.py`
**Gate**: T2.11 is **BLOCKING**.

| ID | Test Name | Runs | Pass Criterion |
|---|---|---|---|
| **T2.1a** | UDE vs Mechanistic (Baseline) | 15,000 | RMSE ≤ 105% of mechanistic |
| **T2.1b** | **UDE vs Mechanistic (Perturbed)** | 15,000 | RMSE ≤ 95% of mechanistic |
| **T2.3** | UKF covariance adaptation | 15,000 | Q-ratio ≥ 1.01 |
| **T2.4** | UKF↔RBPF switching | 12,000 | CRPS improves on switch |
| **T2.8** | 30-min prediction accuracy | 15,000 | RMSE ≤ 15 mg/dL |
| **T2.11** | **Uncertainty Calibration** | 15,000 | **90% PI covers 87-93% of actual** |
| **T2.12** | Adaptation to Shift | 1,000 | RMSE recovers within 48h |

**Perturbation Specs (T2.1b)**:
- Dawn Phenomenon: +20% ISF at 06:00
- Exercise: -30% ISF during activity
- Progressive Drift: +30% ISF over 14 days

---

## 🔍 Layer 3: Causal Inference Engine

**Compute**: ~12-15h | **Files**: `aegis_core/causal_engine.py`

| ID | Test Name | Runs | Pass Criterion |
|---|---|---|---|
| **T3.1** | G-estimation Accuracy | 45,000 | RMSE ≤ 0.10 |
| **T3.3a** | Proximal Inference | 40,000 | Bias reduction ≥ 30% vs no-proxy |
| **T3.3b** | **Weak Instrument Guard** | 5,000 | Bias ≤ 110% when instrument < 0.3 |
| **T3.4** | Confidence Sequence Validity | 100k | Uniform coverage ≥ 93% |
| **T3.7** | **Type I Error Control** | 10,000 | Rejection rate ≤ 5% (Null effect) |
| **T3.8** | Power Analysis | 6,000 | 80% power at N=100 |
| **T3.9** | CS Calibration | 5,000 | 95% CS covers 92-98% |

---

## 🤖 Layer 4: Decision Engine

**Compute**: ~6h | **Files**: `aegis_core/decision_engine.py`

| ID | Test Name | Runs | Pass Criterion |
|---|---|---|---|
| **T4.1** | ACB Variance Reduction | 2,000 | Ratio < 1.0 |
| **T4.2** | **CTS Regret Scaling** | 3,500 | Slope 0.4-0.6 + $R(t)/\sqrt{t} < C$ |
| **T4.3** | Posterior Collapse | 2,500 | Ratio < 0.5 at ≥50% blocking |
| **T4.5** | CTS vs Baselines | 2,000 | Pareto-dominates $\epsilon$-greedy/UCB |
| **T4.10** | Reward Shift Adaptation | 1,500 | Regret recovers within 500 steps |
| **T4.11** | **Negative Control** | 500 | Regret ≈ 0 (identical arms) |

---

## 🛡️ Layer 5: Safety Supervisor

**Status**: ✅ **DONE** (41 Unit Tests + 7 Simulations Passed)
**Files**: `layer5/test_safety_supervisor.py`

Validated safe behavior under:
- Seldonian constraints (0 violations)
- Adversarial attacks (NaN/Inf injections)
- Cold-start scenarios
- Cascading upstream failures

---

## 🌐 Cross-Layer Integration

**Compute**: ~30-40h | **Files**: `integration/test_integration.py`

| ID | Test Name | Runs | Pass Criterion |
|---|---|---|---|
| **CL.1a** | TBR Root-Cause Ablation | 30 × 7d | Ablation identifies responsible layer |
| **CL.1b** | TBR Fix Verification | 30 × 7d | TBR<70 mg/dL ≤ 10% |
| **CL.2a** | AEGIS vs PID Baseline | 15,000 | TIR ≥ PID |
| **CL.2b** | AEGIS vs MPC Baseline | 500 | TBR<54 mg/dL ≤ MPC |
| **CL.5** | **L1→L3 Proxy Chain** | 6,000 | End-to-end bias reduction ≥ 15% |

---

## 📦 Verification Checkpoints

After each layer passes validation, create a git tag:
```bash
git tag l1-validated && git push --tags
git tag l2-validated && git push --tags
# ...
```
