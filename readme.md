<div align="center">

# AEGIS 3.0

### Adaptive Engineering for Generalized Individualized Safety

**A Unified Architecture for Safe, Causal N-of-1 Precision Medicine**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/)

*Bridging the gap between population-level evidence and individual therapeutic response through causal inference, Bayesian state estimation, and formal verification.*

</div>

---

### Note : The Test results that are incldued in the Readme file are just placeholder, for actual test resutls of the architecture refer the test results in each layer.
## 📋 Table of Contents

- [Abstract](#-abstract)
- [Motivation](#-motivation)
  - [The Precision Medicine Challenge](#the-precision-medicine-challenge)
  - [The Small Data Paradox](#the-small-data-paradox)
  - [The Case for Unified Architecture](#the-case-for-unified-architecture)
- [Key Contributions](#-key-contributions)
- [Architecture Overview](#-architecture-overview)
  - [Design Principles](#design-principles)
  - [Five-Layer Architecture](#five-layer-architecture)
  - [Inter-Layer Communication Protocol](#inter-layer-communication-protocol)
- [Layer Specifications](#-layer-specifications)
  - [Layer 1: Semantic Sensorium](#layer-1-semantic-sensorium)
  - [Layer 2: Adaptive Digital Twin](#layer-2-adaptive-digital-twin)
  - [Layer 3: Causal Inference Engine](#layer-3-causal-inference-engine)
  - [Layer 4: Decision Engine](#layer-4-decision-engine)
  - [Layer 5: Simplex Safety Supervisor](#layer-5-simplex-safety-supervisor)
- [Theoretical Foundations](#-theoretical-foundations)
  - [Identification Theorems](#identification-theorems)
  - [Regret Analysis](#regret-analysis)
  - [Safety Guarantees](#safety-guarantees)
- [Experimental Evaluation](#-experimental-evaluation)
  - [Simulation Environment](#simulation-environment)
  - [Baseline Comparisons](#baseline-comparisons)
  - [Main Results](#main-results)
  - [Scenario-Specific Evaluations](#scenario-specific-evaluations)
  - [Ablation Study](#ablation-study)
- [Mathematical Appendix](#-mathematical-appendix)
- [Limitations & Future Directions](#-limitations--future-directions)
- [Ethical Considerations](#-ethical-considerations)
- [Citation](#-citation)
- [References](#-references)
- [License](#-license)

---

## 📄 Abstract

The promise of precision medicine—delivering the right treatment to the right patient at the right time—remains unrealized due to a fundamental epistemological gap between population-derived evidence and individual therapeutic response. The **Average Treatment Effect (ATE)**, the cornerstone of Evidence-Based Medicine, rests on an ergodicity assumption that demonstrably fails in complex biological systems characterized by non-stationarity, path dependence, and feedback dynamics. 

**AEGIS 3.0** (Adaptive Engineering for Generalized Individualized Safety) presents a five-layer unified architecture that synthesizes advances in causal inference, Bayesian state estimation, and formal verification to enable **provably safe, causally valid treatment optimization** for the individual patient.

### Principal Algorithmic Innovations

| Innovation | Description |
|------------|-------------|
| **Proximal G-Estimation with Text-Derived Negative Controls** | Enables causal identification under unmeasured confounding by leveraging semantic features from patient narratives |
| **Adaptive Hybrid State Estimation** | Automatic switching between AC-UKF and RBPF based on detected distributional regime |
| **Counterfactual Thompson Sampling (CTS)** | Novel bandit algorithm maintaining exploration efficiency under hard safety constraints through Digital Twin-imputed posterior updates |
| **Hierarchical Cold-Start Seldonian Constraints** | Population-derived Bayesian priors for initial safety guarantees without patient-specific data |

**Keywords**:  N-of-1 Trials, Causal Inference, Digital Twins, Safe Reinforcement Learning, Precision Medicine, Micro-Randomized Trials, Proximal Causal Inference, Thompson Sampling

---

## 🎯 Motivation

### The Precision Medicine Challenge

For five decades, the Randomized Controlled Trial (RCT) has served as the epistemological gold standard for therapeutic evidence. The statistical validity of applying population-derived conclusions to individual patients rests on an implicit assumption borrowed from statistical mechanics: **ergodicity**—the equivalence of ensemble averages (across patients at one time) and time averages (within one patient across time).

Formally expressed: 

$$\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^{N} Y_i(t) \stackrel{? }{=} \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} Y_i(t)$$

In complex adaptive systems—including human physiology—this equality **demonstrably fails**.  Biological systems exhibit: 

| Phenomenon | Description | Impact on Treatment |
|------------|-------------|---------------------|
| **Hysteresis** | History-dependent responses | Past treatments affect future responses |
| **Non-stationarity** | Time-varying dynamics | Treatment effects change over time |
| **Bifurcations** | Qualitative regime changes | System can shift to entirely different behavioral modes |

> A medication producing a positive Average Treatment Effect (ATE) may be inert, suboptimal, or harmful for a specific individual due to idiosyncratic genetic, environmental, or physiological boundary conditions.

This represents not merely statistical noise to be averaged away, but a **structural inadequacy** of population statistics to characterize individual response. The transition from population-level inference to individual-level optimization constitutes the defining computational challenge of twenty-first century medicine.

### The Small Data Paradox

The **N-of-1 trial**, wherein a single patient serves as their own control across multiple treatment periods, offers a principled solution to the ergodicity problem. However, this design introduces a complementary challenge: the **Small Data Paradox**. 

Modern machine learning achieves its power through massive datasets where the Law of Large Numbers suppresses variance. In N-of-1 trials, we possess perhaps **T=100 observations** for a single individual—insufficient for data-hungry deep learning yet exhibiting complex temporal dependencies that violate classical statistical assumptions.

#### Previous Approaches and Their Failure Modes

| Architecture | Approach | Failure Mode |
|--------------|----------|--------------|
| **VACA-type** | Predictive deep learning (LSTM/RNN) | Confounding by indication; conflated correlation with causation |
| **Discovery-based** | Data-driven causal discovery | Structural instability; hallucinated causal links from sparse data |
| **Standard RL** | Reinforcement learning | Unsafe exploration; sample inefficiency in short trials |

These failures share a common root: attempting to learn complex dynamics *de novo* from radically insufficient data, while ignoring both the rich prior knowledge encoded in physiological science and the safety imperatives of medical intervention.

### The Case for Unified Architecture

Existing approaches address individual challenges—causal inference, state estimation, safe learning, or cold-start—in isolation. However, N-of-1 precision medicine requires their **simultaneous** resolution: 

- A system with perfect causal inference but unsafe exploration will harm patients
- A perfectly safe system that cannot identify causal effects will deliver suboptimal treatment
- A system lacking cold-start capabilities cannot be deployed to new patients

AEGIS 3.0 resolves the Small Data Paradox through a **Grey-Box** architecture that embeds mechanistic physiological priors while learning patient-specific deviations. It addresses the safety imperative through **formal verification** that decouples learning from constraint enforcement.  And it achieves causal validity through **design-based identification** augmented by novel methods for unmeasured confounding adjustment.

---

## 🔬 Key Contributions

AEGIS 3.0 makes three categories of contributions:

### C1: Architectural Integration (Primary)

The first unified architecture that jointly addresses causal identification, state estimation, safe exploration, and cold-start safety for N-of-1 trials.  While individual components draw on existing techniques, their integration is novel and non-trivial—we identify key interface requirements and resolve tensions between competing objectives (e.g., exploration vs. safety). Four design principles guide architectural decisions and enable principled extension.

### C2: Algorithmic Innovations (Secondary)

Four novel algorithmic contributions:

1. **Proximal G-Estimation with Text-Derived Negative Controls**
   - First application of proximal causal inference to N-of-1 trials
   - Uses semantic features from patient narratives as treatment/outcome confounding proxies

2. **Counterfactual Thompson Sampling (CTS)**
   - Bandit algorithm maintaining posterior updates for safety-blocked actions
   - Model-imputed counterfactual outcomes with confidence-weighted likelihood

3. **Hierarchical Cold-Start Seldonian Constraints**
   - Framework for transferring population-level safety posteriors to individual patients
   - Enables probabilistic safety guarantees without patient-specific adverse event data

4. **Adaptive Hybrid State Estimation**
   - Principled criterion for automatic selection between AC-UKF and RBPF
   - Based on detected distributional regime

### C3: Validation Framework (Tertiary)

Comprehensive in-silico validation protocol using the FDA-accepted UVA/Padova Type 1 Diabetes simulator, enabling rigorous evaluation of N-of-1 systems with clinically meaningful endpoints. 

---

## 🏗 Architecture Overview

### Design Principles

The AEGIS 3.0 architecture is guided by four principles derived from the unique challenges of N-of-1 precision medicine:

#### Principle 1: Grey-Box Integration

> Pure data-driven approaches fail with small N-of-1 data (insufficient samples for complex function approximation). Pure mechanistic models cannot capture individual variation (parameters derived from population averages).

AEGIS integrates both: mechanistic priors constrain the hypothesis space to physiologically plausible trajectories while learned components capture patient-specific deviations.

**Architectural Implication**: Layer 2 implements Universal Differential Equations combining fixed physiological models with neural residuals. 

#### Principle 2: Separation of Learning and Safety

> Learning systems must explore to improve; safety systems must be conservative.  Coupling these creates tension that either compromises learning (over-conservative) or safety (under-conservative).

AEGIS decouples them through the Simplex architecture:  the learning system operates freely while an independent verified supervisor enforces constraints.

**Architectural Implication**:  Layers 1-4 learn and optimize; Layer 5 independently verifies and can override any decision.

#### Principle 3: Design-Based Causal Identification

> Observational inference from N-of-1 data is confounded—patients modify behavior based on symptoms, creating treatment-outcome associations that are not causal. 

AEGIS embeds randomization (MRTs) into the decision process, enabling causal identification by design rather than assumption.

**Architectural Implication**: Layer 4 implements micro-randomization with known probabilities; Layer 3 exploits these for unbiased causal estimation.

#### Principle 4: Hierarchical Information Transfer

> Cold-start is inevitable for new patients—no patient-specific data exists on Day 1. Purely patient-specific methods require dangerous exploration periods.

AEGIS transfers population knowledge hierarchically, constraining initial behavior while permitting personalization as data accumulates.

**Architectural Implication**:  All layers maintain hierarchical priors that relax from population to individual as evidence accumulates.

---

### Five-Layer Architecture

AEGIS 3.0 comprises five integrated layers, each addressing a distinct functional requirement while maintaining bidirectional information flow with adjacent layers. 

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AEGIS 3.0 ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    LAYER 5: SIMPLEX SAFETY SUPERVISOR                │     │
│    │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │     │
│    │  │ REFLEX        │  │ STL MONITOR   │  │ SELDONIAN             │   │     │
│    │  │ CONTROLLER    │  │ (Reachability │  │ CONSTRAINTS           │   │     │
│    │  │ (Model-Free)  │  │  Analysis)    │  │ (Hierarchical Prior)  │   │     │
│    │  └───────────────┘  └───────────────┘  └───────────────────────┘   │     │
│    └─────────────────────────────┬───────────────────────────────────────┘     │
│                                  ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    LAYER 4: DECISION ENGINE                          │     │
│    │         COUNTERFACTUAL THOMPSON SAMPLING                            │     │
│    │    ┌──────────────────────────────────────────────────────┐        │     │
│    │    │  • Action-Centered Reward Decomposition               │        │     │
│    │    │  • Posterior Sampling with Safety Filtering           │        │     │
│    │    │  • Counterfactual Updates for Blocked Arms            │        │     │
│    │    └──────────────────────────────────────────────────────┘        │     │
│    └─────────────────────────────┬───────────────────────────────────────┘     │
│                                  ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    LAYER 3: CAUSAL INFERENCE ENGINE                  │     │
│    │    ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐    │     │
│    │    │ HARMONIC       │  │ PROXIMAL       │  │ MARTINGALE       │    │     │
│    │    │ G-ESTIMATION   │  │ ADJUSTMENT     │  │ CONFIDENCE       │    │     │
│    │    │ (Circadian)    │  │ (Unmeasured U) │  │ SEQUENCES        │    │     │
│    │    └────────────────┘  └────────────────┘  └──────────────────┘    │     │
│    │                    INDIVIDUAL TREATMENT EFFECT τ(S_t)              │     │
│    └─────────────────────────────┬───────────────────────────────────────┘     │
│                                  ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    LAYER 2: ADAPTIVE DIGITAL TWIN                    │     │
│    │    ┌──────────────────────────────────────────────────────────┐    │     │
│    │    │           UNIVERSAL DIFFERENTIAL EQUATION                 │    │     │
│    │    │     dx/dt = f_mech(x, u; θ_fixed) + NN(x, u; θ_learn)   │    │     │
│    │    └──────────────────────────────────────────────────────────┘    │     │
│    │    ┌─────────────┐    SWITCHING    ┌─────────────────┐            │     │
│    │    │  AC-UKF     │ ◄───CRITERION───►│     RBPF        │            │     │
│    │    │ (Gaussian)  │                  │  (Multimodal)   │            │     │
│    │    └─────────────┘                  └─────────────────┘            │     │
│    │                    HIDDEN STATE ESTIMATE x̂_t ± P_t                 │     │
│    └─────────────────────────────┬───────────────────────────────────────┘     │
│                                  ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    LAYER 1: SEMANTIC SENSORIUM                       │     │
│    │    ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐    │     │
│    │    │ ONTOLOGY-      │  │ PROBABILISTIC  │  │ CAUSAL ROLE      │    │     │
│    │    │ CONSTRAINED    │  │ TEMPORAL       │  │ CLASSIFICATION   │    │     │
│    │    │ EXTRACTION     │  │ GROUNDING      │  │ (Z_t, W_t)       │    │     │
│    │    └────────────────┘  └────────────────┘  └──────────────────┘    │     │
│    │              SEMANTIC ENTROPY FILTER (HITL Trigger)                │     │
│    └─────────────────────────────┬───────────────────────────────────────┘     │
│                                  ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │    RAW DATA:   Wearables │ CGM │ PRO Surveys │ Patient Diaries       │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Inter-Layer Communication Protocol

Information flows bidirectionally through the architecture:

#### Upward Flow (Inference)

| From → To | Data Transmitted |
|-----------|------------------|
| Layer 1 → Layer 2 | Structured observations $(S_t, Y_t)$ and proxy variables $(Z_t, W_t)$ |
| Layer 2 → Layer 3 | State estimates $\hat{x}_t$ with uncertainty $P_t$ |
| Layer 3 → Layer 4 | Treatment effect estimates $\hat{\tau}(S_t)$ with confidence bounds |
| Layer 4 → Layer 5 | Proposed action $A_t^{proposed}$ for safety verification |

#### Downward Flow (Control)

| From → To | Data Transmitted |
|-----------|------------------|
| Layer 5 → Layer 4 | Safety-certified action $A_t^{safe}$ or blocking signal |
| Layer 4 → Layer 3 | Randomization probability $p_t(S_t)$ for causal inference |
| Layer 3 → Layer 2 | Causal constraints for counterfactual simulation |
| Layer 2 → Layer 1 | Expected observations for anomaly detection |

---

## 📐 Layer Specifications

### Layer 1: Semantic Sensorium

#### Problem Statement

N-of-1 digital trials generate heterogeneous data streams:  continuous sensor measurements, periodic surveys, and unstructured patient narratives. The data layer must accomplish three objectives:

1. **Semantic Standardization**: Map diverse inputs to a consistent clinical ontology
2. **Uncertainty Quantification**:  Detect and flag unreliable extractions
3. **Causal Proxy Identification**: Extract variables suitable for confounding adjustment

#### Ontology-Constrained Extraction

AEGIS 3.0 enforces semantic consistency through **constrained generation**.  Rather than extracting free-form text, the extraction module maps patient narratives to SNOMED-CT concept identifiers through grammatically constrained decoding.  This ensures that semantically equivalent expressions ("drowsy," "sleepy," "tired," "zonked out") map to identical nodes in the causal graph, preventing artificial sparsity.

**Extraction Output Schema:**

```python
@dataclass
class Observation:
    concept_id: str      # SNOMED-CT Identifier
    value: Union[float, str]  # Numeric or Categorical
    unit: str            # UCUM Standard Unit
    timestamp: datetime  # ISO-8601 with mandatory timezone
    confidence: float    # Range:  [0, 1]
    semantic_entropy: float  # Range: [0, ∞)
```

#### Semantic Entropy Thresholding

Standard confidence scores fail to capture *semantic* uncertainty. A model may assign 95% probability to an extraction while being fundamentally uncertain about its meaning.  AEGIS 3.0 implements **Semantic Entropy** quantification: 

1. Generate K candidate extractions with varying sampling temperatures
2. Embed candidates in SNOMED-CT semantic space
3. Cluster candidates by semantic equivalence (same concept ID)
4. Compute entropy over cluster distribution: 

$$H_{sem}(\mathcal{T}_t) = -\sum_{c \in \mathcal{C}} p(c) \log p(c)$$

where $p(c)$ is the proportion of candidates falling in semantic cluster $c$. 

**Decision Rule**:  Trigger Human-in-the-Loop (HITL) review when $H_{sem} > \delta_{entropy}$, indicating semantically distinct interpretations with non-trivial probability.

#### Causal Role Classification for Proximal Inference

A principal innovation of AEGIS 3.0 is leveraging patient narratives as sources of **negative control proxies** for unmeasured confounding adjustment. 

> **Definition (Treatment-Confounder Proxy)**: A variable $Z_t$ extracted from text serves as a valid treatment-confounder proxy if: 
> - $Z_t \perp\!\!\!\perp Y_t \mid U_t, S_t$ (no direct effect on outcome)
> - $Z_t \not\perp\!\!\!\perp U_t \mid S_t$ (associated with unmeasured confounder)
> - $Z_t \perp\!\!\!\perp A_t \mid U_t, S_t$ (not caused by treatment)

> **Definition (Outcome-Confounder Proxy)**: A variable $W_t$ serves as a valid outcome-confounder proxy if:
> - $W_t \perp\!\!\!\perp A_t \mid U_t, S_t$ (not caused by treatment)
> - $W_t \not\perp\!\!\!\perp U_t \mid S_t$ (associated with unmeasured confounder)

**Example**:  Consider unmeasured psychological stress ($U_t$) affecting both medication adherence ($A_t$) and symptom severity ($Y_t$):
- Patient diary mentions of "work deadline" ($Z_t$) may serve as treatment-proxy (stress causes deadline mention; deadline doesn't directly affect symptoms)
- Mentions of "couldn't sleep" ($W_t$) may serve as outcome-proxy (stress causes poor sleep; poor sleep predicts symptoms but isn't caused by today's treatment)

---

### Layer 2: Adaptive Digital Twin

#### Universal Differential Equations

The Digital Twin maintains a dynamic model of patient physiology through **Universal Differential Equations (UDEs)**:

$$\frac{dx}{dt} = f_{mech}(x, u; \theta_{fixed}) + f_{NN}(x, u; \theta_{learned})$$

where:
- $f_{mech}$ encodes established physiological mechanisms (e.g., insulin-glucose dynamics via the Bergman Minimal Model)
- $f_{NN}$ is a neural network learning patient-specific deviations from textbook physiology
- $\theta_{fixed}$ are literature-derived parameters
- $\theta_{learned}$ are personalized parameters estimated from patient data

This architecture resolves the Small Data Paradox:  the mechanistic prior constrains the hypothesis space to physiologically plausible trajectories, while the neural residual captures individual variation.

#### Adaptive Constrained UKF (AC-UKF)

For unimodal state distributions, AEGIS 3.0 implements the **Adaptive Constrained UKF** with two innovations:

**Innovation-Based Covariance Adaptation**:  The filter monitors measurement residuals $\epsilon_k = y_k - h(\hat{x}_k^-)$.  If empirical residual variance exceeds theoretical prediction, process noise covariance $Q_k$ is inflated: 

$$Q_{k+1} = Q_k + \alpha K_k \left( \epsilon_k \epsilon_k^T - S_k \right) K_k^T$$

where $S_k$ is the predicted residual covariance and $K_k$ is the Kalman gain. 

**Constraint Projection**: Before propagating sigma points through the ODE, a projection operator enforces physiological constraints:

$$\mathcal{X}_{sigma}^{proj} = \Pi_{\mathcal{C}}(\mathcal{X}_{sigma})$$

This prevents numerical instabilities from unphysical states.

#### Rao-Blackwellized Particle Filter (RBPF)

When state distributions become multimodal—during regime transitions, disease exacerbations, or bifurcation events—Gaussian approximations fail categorically.  AEGIS 3.0 employs **RBPF** for such regimes. 

RBPF exploits conditional linearity:  partition states into $x = [x_{lin}, x_{nl}]$ where linear dynamics govern $x_{lin}$ conditional on $x_{nl}$. The posterior factorizes: 

$$p(x_{lin}, x_{nl} \mid y_{1:t}) = p(x_{lin} \mid x_{nl}, y_{1:t}) \cdot p(x_{nl} \mid y_{1:t})$$

The linear component admits closed-form Kalman updates; only the nonlinear component requires particle approximation.

#### Automatic Filter Selection

AEGIS 3.0 implements automatic switching based on distribution diagnostics:

**Switching Criterion**: At each timestep, evaluate:
1. **Normality Test**: Shapiro-Wilk statistic on recent residuals
2. **Bimodality Coefficient**: $BC = \frac{skewness^2 + 1}{kurtosis}$

**Decision Rule**: 

| Condition | Action |
|-----------|--------|
| Shapiro-Wilk $p < 0.05$ OR $BC > 0.555$ | Deploy RBPF (non-Gaussian/multimodal detected) |
| Otherwise | Deploy AC-UKF (Gaussian adequate) |
| RBPF effective sample size below threshold | Trigger resampling |

---

### Layer 3: Causal Inference Engine

#### Micro-Randomized Trial Design

At each decision point $k$, treatment $A_k$ is randomized with probability $p_k(S_k)$ conditional on observed context.  This design maximizes effective sample size while maintaining causal identification through known randomization probabilities.

**Positivity Constraint**: $\epsilon < p_k(S_k) < 1 - \epsilon$ for all contexts, ensuring all treatment-context combinations remain possible.

#### Harmonic Time-Varying G-Estimation

Standard G-estimation assumes time-invariant treatment effects. AEGIS 3.0 implements **Harmonic G-Estimation** with time-varying effects:

**Baseline Model** (Fourier decomposition):

$$\mu(t; \beta) = \beta_0 + \sum_{k=1}^{K} \left[ \beta_{ck} \cos\left(\frac{2\pi k t}{24}\right) + \beta_{sk} \sin\left(\frac{2\pi k t}{24}\right) \right]$$

**Treatment Effect Model** (time-varying):

$$\tau(t; \psi) = \psi_0 + \sum_{k=1}^{K} \left[ \psi_{ck} \cos\left(\frac{2\pi k t}{24}\right) + \psi_{sk} \sin\left(\frac{2\pi k t}{24}\right) \right]$$

**Estimating Equation**:

$$\sum_{t=1}^{T} \left[ Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi) A_t \right] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t) = 0$$

This formulation allows treatment effects to **vary by time of day** while orthogonalizing against circadian baseline variation.

#### Double Robustness Property

> **Theorem (Double Robustness)**: Under positivity and consistency assumptions, the Harmonic G-estimator $\hat{\psi}$ converges in probability to the true effect $\psi^*$ if either:
> 1. $\hat{\mu}(S_t) = \mathbb{E}[Y_{t+1} \mid S_t, A_t=0]$ (outcome model correctly specified), OR
> 2. $p_t(S_t) = \mathbb{P}(A_t=1 \mid S_t)$ (propensity model correctly specified)

Since randomization probabilities are determined algorithmically in MRTs, condition (2) is satisfied by construction.

#### Proximal G-Estimation for Unmeasured Confounding

When unmeasured confounders $U_t$ violate sequential ignorability, standard G-estimation produces biased effect estimates. AEGIS 3.0 integrates **Proximal Causal Inference** using text-derived negative controls. 

> **Assumption (Proxy Completeness)**: The treatment-confounder proxy $Z_t$ and outcome-confounder proxy $W_t$ satisfy:
> $$\text{span}\{\mathbb{E}[h(W) \mid Z, S]\} = L^2(U \mid S)$$

Under this richness condition, a **Bridge Function** $h^*(W_t)$ exists such that adjustment recovers the causal effect despite $U_t$ being unobserved. 

**Augmented Estimating Equation**: 

$$\sum_{t=1}^{T} \left[ Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi) A_t - h^*(W_t) \right] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t) = 0$$

> **Theorem (Proximal Identification)**: Under proxy completeness and standard regularity conditions, the proximal G-estimator identifies the causal effect $\psi^*$ even when $U_t \not\in H_t$.

#### Anytime-Valid Inference

Adaptive trials require **continuous monitoring** without inflating Type-I error.  AEGIS 3.0 employs **Martingale Confidence Sequences** that maintain coverage guarantees at arbitrary stopping times.

> **Definition (Confidence Sequence)**: A sequence of confidence sets $\{CS_t\}_{t=1}^{\infty}$ is $(1-\alpha)$-valid if:
> $$\mathbb{P}\left( \psi^* \in CS_t \text{ for all } t \geq 1 \right) \geq 1 - \alpha$$

---

### Layer 4: Decision Engine

#### Action-Centered Contextual Bandits

Standard reinforcement learning attempts to learn the total reward function $Q(S, A)$. In N-of-1 trials, reward variance is dominated by baseline health fluctuations unrelated to treatment.  AEGIS 3.0 employs **Action-Centered Bandits** that decompose reward: 

$$R_t = f(S_t) + A_t \cdot \tau(S_t) + \epsilon_t$$

The bandit learns *only* $\tau(S_t)$—the treatment effect—treating $f(S_t)$ as noise to be subtracted.  This **variance reduction** accelerates learning by orders of magnitude.

> **Theorem (Regret Bound)**: The Action-Centered Bandit achieves regret: 
> $$\mathcal{R}(T) = \tilde{O}(d_{\tau} \sqrt{T})$$
> where $d_{\tau}$ is the dimension of treatment effect parameters.

#### Counterfactual Thompson Sampling (CTS)

Standard constrained bandits create a **pathology**:  if the optimal action lies near the safety boundary, it may be repeatedly blocked. The posterior for this action never updates—**posterior collapse**—leaving the system uncertain about potentially excellent treatments indefinitely.

AEGIS 3.0 introduces **Counterfactual Thompson Sampling (CTS)**:

```
Algorithm:  Counterfactual Thompson Sampling
─────────────────────────────────────────────────────────────────
Input: Posterior P(θ | H_t), safety evaluator S, Digital Twin D

1. SAMPLE:   Draw θ̃ ~ P(θ | H_t)

2. OPTIMIZE:   Compute unconstrained optimum
              a* = argmax_a E[R | S_t, a, θ̃]

3. SAFETY CHECK:  Query safety supervisor for a*
   - If S(a*, S_t) = SAFE: Execute A_t = a*
   - If S(a*, S_t) = UNSAFE:  Proceed to Step 4

4. COUNTERFACTUAL UPDATE (for blocked action a*):
   - Impute counterfactual outcome:   Ŷ_{a*} = D. predict(S_t, a*)
   - Compute imputation confidence: λ = D.confidence(S_t, a*)
   - Update posterior with discounted likelihood: 
     P(θ | H_{t+1}) ∝ P(Ŷ_{a*} | θ, S_t, a*)^λ · P(θ | H_t)

5. SAFE SELECTION: Execute A_t = argmax_{a ∈ A_safe} E[R | S_t, a, θ̃]
─────────────────────────────────────────────────────────────────
```

**Key Innovation**: Step 4 updates the posterior for the blocked action using Digital Twin predictions.  The discount factor $\lambda \in (0, 1)$ reflects imputation uncertainty—high confidence yields stronger updates; low confidence yields weak updates.

> **Theorem (CTS Regret Bound)**: Under bounded rewards, accurate safety constraints, and bounded imputation error, CTS achieves: 
> $$\mathcal{R}(T) \leq \tilde{O}(d_{\tau} \sqrt{T \log T}) + O(B_T \cdot \Delta_{max} \cdot (1-\lambda))$$
> where $B_T$ is the number of blocking events and $\Delta_{max}$ is the maximum suboptimality gap.

---

### Layer 5: Simplex Safety Supervisor

#### Three-Tier Safety Hierarchy

AEGIS 3.0 implements three safety tiers with strict priority ordering:

| Tier | Component | Mechanism | Example |
|------|-----------|-----------|---------|
| **1** (Highest) | Reflex Controller | Model-free threshold logic on raw sensors | "If glucose < 55 mg/dL, halt all insulin recommendations" |
| **2** | STL Monitor | Formal verification via reachability analysis | $\Box_{[0,T]}(G > 70) \wedge \Box_{[0,T]}(G < 250)$ |
| **3** | Seldonian Constraints | High-confidence probabilistic bounds | $\mathbb{P}(g(\theta) > 0) \leq \alpha$ |

**Conflict Resolution**: When tiers disagree, higher-priority tier prevails. 

#### Breaking the Circularity Problem

Previous approaches suffered from **safety circularity**: the STL monitor relied on Digital Twin predictions; if the Twin diverged, safety checks became meaningless. 

AEGIS 3.0 breaks this circularity through **Reachability Analysis** using population-derived worst-case bounds independent of the patient-specific Digital Twin: 

> **Definition (Conservative Physiological Bounds)**: For physiological variable $x$, define: 
> - Maximum rate of change: $|\dot{x}| \leq \dot{x}_{max}$ (from population studies)
> - Action delay bounds: $t_{onset} \in [t_{min}, t_{max}]$, $t_{peak} \in [t_{min}', t_{max}']$
> - Physiological limits: $x \in [x_{min}, x_{max}]$

**Reachability Set**: For current state $x_t$ and proposed action $a_t$, compute worst-case future states: 

$$\mathcal{R}_{t+\Delta}(x_t, a_t) = \{ x' :  \exists \text{ trajectory from } x_t \text{ under } a_t \text{ respecting bounds} \}$$

**Safety Decision**:

$$A_{final} = \begin{cases}
A_{complex} & \text{if } \mathcal{R}_{t+\Delta} \cap \mathcal{X}_{unsafe} = \emptyset \\
A_{reflex} & \text{otherwise}
\end{cases}$$

#### Cold Start Safety via Hierarchical Priors

On Day 1, no patient-specific safety data exists. AEGIS 3.0 implements **Hierarchical Bayesian Prior Transfer**:

**Population Model** (from historical RCTs and registries):

$$\theta_{pop} \sim \mathcal{N}(\mu_0, \Lambda_0^{-1})$$
$$\Sigma_{between} \sim \text{Inverse-Wishart}(\nu_0, \Psi_0)$$

**Individual Model** (Day 1, no data):

$$\theta_i \mid \theta_{pop} \sim \mathcal{N}(\theta_{pop}, \Sigma_{between})$$

**Day 1 Safety Bound**: Use conservative tail of population distribution: 

$$\theta_{safe} = \theta_{pop} - z_{\alpha_{strict}} \cdot \sqrt{\text{diag}(\Sigma_{between})}$$

where $\alpha_{strict} = 0.01$ (99% safe in population).

**Relaxation Schedule**: As patient data accumulates, transition from population to individual posterior:

$$\alpha_t = \alpha_{strict} \cdot e^{-t/\tau} + \alpha_{standard} \cdot (1 - e^{-t/\tau})$$

where $\tau$ controls relaxation rate (typically 10-14 days) and $\alpha_{standard} = 0.05$. 

---

## 📊 Theoretical Foundations

### Identification Theorems

#### Theorem 1: Harmonic G-Estimation Identification

Under the assumptions of: 
1. **Consistency**: $Y_t = Y_t^{a}$ when $A_t = a$
2. **Positivity**: $\epsilon < p_t(S_t) < 1 - \epsilon$ for all $t, S_t$
3. **Sequential Ignorability**: $Y_{t+1}^{\bar{a}} \perp\!\!\!\perp A_t \mid H_t$

The Harmonic G-estimator identifies the time-varying causal effect: 

$$\tau(t; \psi^*) = \mathbb{E}[Y_{t+1}^{1} - Y_{t+1}^{0} \mid S_t, t]$$

#### Theorem 2: Proximal Identification

When sequential ignorability fails due to unmeasured confounder $U_t$, but valid proxies $(Z_t, W_t)$ exist satisfying proxy completeness, the Proximal G-estimator identifies the causal effect $\psi^*$ even when $U_t \not\in H_t$.

#### Theorem 3: Double Robustness

The combined estimator is consistent if either:
1. The Digital Twin correctly specifies $\mathbb{E}[Y_{t+1} \mid S_t, A_t = 0]$, OR
2. The randomization probabilities $p_t(S_t)$ are correctly specified (true by design in MRTs)

### Regret Analysis

#### Theorem 4: Safe Exploration Regret

Under the AEGIS 3.0 architecture with CTS, total regret satisfies:

$$\mathcal{R}(T) \leq \underbrace{O(d_{\tau} \sqrt{T \log T})}_{\text{Learning regret}} + \underbrace{O(B_T \cdot \Delta_{max} \cdot (1-\lambda))}_{\text{Safety blocking regret}}$$

where:
- $d_{\tau}$ = dimension of treatment effect parameters
- $B_T$ = number of safety-blocked decisions
- $\Delta_{max}$ = maximum suboptimality of safe alternatives
- $\lambda$ = average imputation confidence

### Safety Guarantees

#### Theorem 5: Simplex Safety

Under the Simplex architecture with reachability analysis using valid conservative bounds: 

$$\mathbb{P}(\text{Safety Violation}) = 0$$

for all constraints expressible in STL with known physiological bounds.

#### Theorem 6: Cold Start Safety

Under the hierarchical prior with $\alpha_{strict} = 0.01$:

$$\mathbb{P}(\text{Day 1 Safety Violation}) \leq 0.01$$

with probability converging to patient-specific $\alpha_{standard} = 0.05$ as $t \to \infty$. 

---

## 🧪 Experimental Evaluation

### Simulation Environment

We evaluate AEGIS 3.0 using the **UVA/Padova Type 1 Diabetes Simulator** (simglucose), an FDA-accepted simulator for testing insulin dosing algorithms. 

| Parameter | Value |
|-----------|-------|
| **Virtual Patients** | N = 30 (10 children, 10 adolescents, 10 adults) |
| **Trial Duration** | 8 weeks per patient (2 weeks run-in, 4 weeks active, 2 weeks washout) |
| **Decision Points** | 6 per day (meal times and between-meal periods) |
| **Intervention** | Bolus timing suggestions and activity prompts |
| **Randomization** | MRT with contextual stratification |
| **Random Seeds** | 5 per configuration for statistical power |

### Baseline Comparisons

| System | Description |
|--------|-------------|
| **Standard PID Control** | Classical proportional-integral-derivative controller with population-tuned parameters |
| **Naive RL** | Q-learning with linear function approximation and ε-greedy exploration |
| **JITAI (No Causal)** | Just-in-time adaptive intervention with regression-based effect estimation |
| **Digital Twin Only** | UDE-based prediction with greedy action selection |

### Main Results

#### End-to-End Performance Comparison

| Method | TIR (%) ↑ | Hypo Events ↓ | Hyper Events ↓ | Safety Violations ↓ |
|--------|-----------|---------------|----------------|---------------------|
| Standard PID | 62.3 ± 4.2 | 3.2 ± 1.1 | 5.4 ± 1.8 | 2.1 ± 0.8 |
| Naive RL | 58.7 ± 6.3 | 4.8 ± 2.3 | 4.1 ± 1.5 | 5.3 ± 2.1 |
| JITAI (No Causal) | 67.1 ± 3.8 | 2.4 ± 0.9 | 4.8 ± 1.4 | 1.4 ± 0.6 |
| Digital Twin Only | 69.4 ± 3.5 | 2.1 ± 0.8 | 3.9 ± 1.2 | 0.8 ± 0.4 |
| **AEGIS 3.0** | **78.2 ± 2.9** | **1.2 ± 0.5** | **2.3 ± 0.8** | **0.0 ± 0.0** |

*Results averaged over 30 virtual patients, 8-week trials, 5 random seeds.  Bold indicates statistically significant improvement (p < 0.01, paired t-test with Bonferroni correction).*

**Key Finding**: AEGIS 3.0 achieves **25. 5% relative improvement** in time-in-range over standard PID control and **16.5% improvement** over the best baseline (Digital Twin Only), with **zero safety violations**.

### Scenario-Specific Evaluations

#### Scenario A: Non-Stationarity ("Flu Shock")

Simulated sudden physiological shift (insulin sensitivity increases 50% at t=24h, simulating acute illness onset).

```
              Glucose (mg/dL)
       200 ┤
           │     ╭───╮
       180 ┤    ╱     ╲         ── True
           │   ╱       ╲        ── AEGIS (AC-UKF adaptive)
       160 ┤  ╱         ╲       ·· Standard UKF
           │ ╱           ╲
       140 ┤╱   [Flu shock at t=24h]
           │                ╲
       120 ┤                 ╲───────
           │                     ↑
       100 ┤                   AEGIS adapts
           │                   within 6h
        80 ┼────┬────┬────┬────┬────┬────
           0   12   24   36   48   60   72  Hours
```

| Metric | Standard UKF | AC-UKF (AEGIS) | Oracle |
|--------|--------------|----------------|--------|
| RMSE (mg/dL) | 34.2 ± 8.1 | 18.4 ± 4.3 | 12.1 ± 2.8 |
| Time-to-Adapt (h) | >24 | 5.8 ± 1.2 | — |
| Effect Bias | 0.42 ± 0.15 | 0.08 ± 0.04 | 0.00 |

#### Scenario B:  Circadian Confounding ("Time-of-Day Trap")

Ground truth: τ(morning) = 0.0, τ(evening) = 0.5. Patient availability biased toward evening (70% evening, 30% morning).

| Metric | Naive MRT | Time-Stratified | Harmonic (AEGIS) |
|--------|-----------|-----------------|------------------|
| Abs Bias (ψ₀) | 0.31 ± 0.08 | 0.18 ± 0.06 | 0.04 ± 0.02 |
| 95% CI Coverage | 0.72 | 0.86 | 0.94 |

#### Scenario C:  Exploration Collapse ("Seldonian Bottleneck")

Ground truth:  Optimal action has risk = 5. 5%, safety threshold = 5.0%. 

| Metric | Standard TS | ε-Greedy | Conservative | CTS (AEGIS) |
|--------|-------------|----------|--------------|-------------|
| Regret (T=1000) | 98.2 ± 12.4 | 67.3 ± 9.1 | 45.2 ± 6.8 | **28.4 ± 4.2** |
| Safety Violations | 0. 0 | 2.1 ± 0.8 | 0.0 | 0.0 |
| Posterior Var (blocked) | 0.89 | 0.76 | 0.91 | 0.24 |

### Ablation Study

| Configuration | TIR (%) | Δ from Full | Safety Viol. |
|---------------|---------|-------------|--------------|
| **Full AEGIS 3.0** | 78.2 | — | 0.0 |
| − Proximal Adjustment | 74.1 | -4.1 | 0.0 |
| − Harmonic G-Estimation | 73.5 | -4.7 | 0.0 |
| − Counterfactual TS | 71.8 | -6.4 | 0.0 |
| − Adaptive Filtering | 75.6 | -2.6 | 0.2 |
| − Cold-Start Safety | 76.4* | -1.8 | 1.3* |
| − Simplex Safety | 72.3† | -5.9 | 4.2† |

*\* Similar asymptotic TIR but 1. 3 safety violations concentrated in first week*
*† 4.2 safety violations throughout trial*

### Computational Feasibility

| Component | Time Complexity | Mean Runtime | Max Runtime |
|-----------|-----------------|--------------|-------------|
| Semantic Extraction | O(L·V) | 0.32s | 0.48s |
| AC-UKF Update | O(n³) | 0.008s | 0.012s |
| RBPF Update | O(N·n²) | 0.21s | 0.38s |
| G-Estimation | O(T·K²) | 12.4s (batch) | 18.2s |
| CTS Selection | O(\|A\|·d) | 0.04s | 0.07s |
| STL Monitoring | O(T·|φ|) | 0.02s | 0.03s |

*Runtime measured on standard computing hardware (Intel i7, 16GB RAM). Parameters:  n=6 states, N=500 particles, L=512 tokens, V=32000 vocabulary, T=1000 timepoints, K=3 harmonics, |A|=5 actions, d=10 parameters.*

All time-critical components (safety monitoring, action selection) complete within clinically acceptable timescales.  Batch operations (G-estimation) run asynchronously without affecting decision latency. 

---

## 📐 Mathematical Appendix

### Proof of Theorem 1: Harmonic G-Estimation Identification

Under the structural nested mean model: 

$$\mathbb{E}[Y_{t+1} \mid H_t, A_t] = \mu(S_t) + \tau(t; \psi^*) \cdot A_t$$

The estimating equation is:

$$U_T(\psi) = \sum_{t=1}^{T} [Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi) A_t] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t)$$

Taking expectation under the true $\psi^*$:

$$\mathbb{E}[U_T(\psi^*)] = \sum_{t=1}^{T} \mathbb{E}\left[ \mathbb{E}[Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi^*) A_t \mid H_t, A_t] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t) \right]$$

Under correct specification of either $\hat{\mu}$ or $p_t$: 

$$\mathbb{E}[Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi^*) A_t \mid H_t, A_t = a] = \mu(S_t) - \hat{\mu}(S_t)$$

If $\hat{\mu}(S_t) = \mu(S_t)$, the inner expectation is zero. 

Alternatively, if $p_t(S_t) = \mathbb{P}(A_t = 1 \mid H_t)$ (true by MRT design):

$$\mathbb{E}[(A_t - p_t(S_t)) \mid H_t] = p_t(S_t) - p_t(S_t) = 0$$

By iterated expectations, $\mathbb{E}[U_T(\psi^*)] = 0$, establishing unbiasedness. 

The Fourier basis $\mathbf{h}(t) = [1, \cos(2\pi t/24), \sin(2\pi t/24), \ldots]^T$ spans the space of smooth periodic functions with period 24 hours, capturing circadian variation in treatment effects.

By standard M-estimation theory, the solution $\hat{\psi}$ to $U_T(\hat{\psi}) = 0$ converges in probability to $\psi^*$ as $T \to \infty$.  ∎

---

### Proof of Theorem 2: Proximal Identification

The causal DAG includes unmeasured confounder $U_t$ affecting both $A_t$ and $Y_{t+1}$:

$$U_t \to A_t, \quad U_t \to Y_{t+1}$$

The treatment-confounder proxy $Z_t$ and outcome-confounder proxy $W_t$ satisfy: 
1. $Z_t \perp\!\!\!\perp Y_t \mid U_t, S_t$ (Z does not directly affect Y)
2. $W_t \perp\!\!\!\perp A_t \mid U_t, S_t$ (W is not affected by A)
3. Both $Z_t, W_t$ are associated with $U_t$ conditional on $S_t$

Under the completeness assumption: 

$$\text{span}\{\mathbb{E}[h(W) \mid Z, S]\} = L^2(U \mid S)$$

there exists a bridge function $h^*(W_t)$ such that: 

$$\mathbb{E}[h^*(W_t) \mid Z_t, S_t] = \mathbb{E}[U_t \mid Z_t, S_t]$$

The confounding bias in standard G-estimation is: 

$$\text{Bias} = \mathbb{E}[Y_{t+1} \mid A_t, S_t] - \mathbb{E}[Y_{t+1} \mid \text{do}(A_t), S_t] = \gamma \cdot \mathbb{E}[U_t \mid A_t, S_t]$$

where $\gamma$ is the effect of $U_t$ on $Y_{t+1}$.

The augmented estimating equation includes the bridge function $h^*(W_t)$ which absorbs the confounding bias:

$$\mathbb{E}[h^*(W_t) \mid A_t, S_t] \approx \gamma \cdot \mathbb{E}[U_t \mid A_t, S_t]$$

Under completeness, this adjustment removes the bias from unmeasured confounding, and the estimator identifies the causal effect $\psi^*$. 

The bridge function is estimated via kernel ridge regression on the joint distribution $(Z_t, W_t, Y_t, A_t, S_t)$, leveraging the proxy structure to integrate out the unobserved confounder.  ∎

---

### Proof of Theorem 4: CTS Regret Bound

Decompose regret into two components:

**Component 1: Learning Regret**

For rounds where the optimal action is not blocked, standard Thompson Sampling analysis applies.  Following Russo & Van Roy (2014), the Bayesian regret of Thompson Sampling with $d_\tau$-dimensional linear reward model is:

$$\mathcal{R}_1(T) \leq O(d_{\tau} \sqrt{T \log T})$$

**Component 2: Blocking Regret**

Let $B_T$ denote the number of rounds where the optimal action $a^*_t$ is blocked by safety constraints. In each such round, the agent executes a suboptimal safe action with gap $\Delta_t \leq \Delta_{max}$. 

Without counterfactual updates, blocking regret would be:

$$\mathcal{R}_2^{naive}(T) = \sum_{t:  \text{blocked}} \Delta_t \leq B_T \cdot \Delta_{max}$$

With counterfactual updates, the posterior for blocked action $a^*$ is updated using imputed outcome $\hat{Y}_{a^*}$ with confidence $\lambda$.  The effective information gain from blocking round $t$ is: 

$$I_t^{CF} = \lambda \cdot I_t^{actual}$$

where $I_t^{actual}$ is the information gain from an actual observation. 

This reduces the probability that the same action is blocked in future rounds.  The effective blocking count is:

$$B_T^{eff} = B_T \cdot (1 - \lambda)$$

yielding blocking regret:

$$\mathcal{R}_2(T) \leq B_T \cdot \Delta_{max} \cdot (1 - \lambda)$$

Combining both components:

$$\mathcal{R}(T) = \mathcal{R}_1(T) + \mathcal{R}_2(T) \leq O(d_{\tau} \sqrt{T \log T}) + O(B_T \cdot \Delta_{max} \cdot (1-\lambda))$$

As Digital Twin accuracy improves ($\lambda \to 1$), the blocking regret vanishes, and CTS achieves optimal $\tilde{O}(\sqrt{T})$ regret.  ∎

---

### Proof of Theorem 5: Simplex Safety

The safety specification is expressed in Signal Temporal Logic (STL):

$$\phi_{safety} = \Box_{[0,T]}(x \in \mathcal{X}_{safe})$$

where $\mathcal{X}_{safe}$ is the safe region of state space.

**Step 1: Conservative Bound Validity**

The reachability analysis uses population-derived bounds: 
- $|\dot{x}| \leq \dot{x}_{max}$:  Maximum physiological rate of change
- Action effects bounded:  $\Delta x \in [\Delta_{min}, \Delta_{max}]$

These bounds are derived from population studies and represent physiological constraints that hold for all patients.

**Step 2: Reachability Set Construction**

For current state $x_t$ and proposed action $a_t$, the reachability set is:

$$\mathcal{R}_{t+\Delta}(x_t, a_t) = \{x' :  \exists \text{ trajectory from } x_t \text{ to } x' \text{ under } a_t \text{ respecting bounds}\}$$

By construction, this set overapproximates all possible trajectories: 

$$\forall \text{ actual trajectory } x(\cdot): x(t+\Delta) \in \mathcal{R}_{t+\Delta}(x_t, a_t)$$

**Step 3: Safety Decision**

The Simplex supervisor approves action $a_t$ only if: 

$$\mathcal{R}_{t+\Delta}(x_t, a_t) \cap \mathcal{X}_{unsafe} = \emptyset$$

If this condition fails, the reflex controller executes a conservative default action $a_{reflex}$ with guaranteed safety. 

**Step 4: Safety Guarantee**

Since the reachability set overapproximates all possible trajectories: 
- If $\mathcal{R}_{t+\Delta} \cap \mathcal{X}_{unsafe} = \emptyset$, no trajectory can reach unsafe states
- If $\mathcal{R}_{t+\Delta} \cap \mathcal{X}_{unsafe} \neq \emptyset$, the action is blocked

Therefore, no approved action can lead to safety violation:  $\mathbb{P}(\text{Safety Violation}) = 0$. ∎

---

### Proof of Theorem 6: Cold Start Safety

**Step 1: Hierarchical Model**

- Population parameters: $\theta_{pop} \sim \mathcal{N}(\mu_0, \Lambda_0^{-1})$
- Between-patient variance: $\Sigma_{between} \sim \text{Inverse-Wishart}(\nu_0, \Psi_0)$
- Individual parameters: $\theta_i \mid \theta_{pop}, \Sigma_{between} \sim \mathcal{N}(\theta_{pop}, \Sigma_{between})$

**Step 2: Day 1 Action Restriction**

On Day 1, no patient-specific data exists.  The safety bound uses the conservative tail: 

$$\theta_{safe} = \theta_{pop} - z_{\alpha_{strict}} \cdot \sqrt{\text{diag}(\Sigma_{between})}$$

where $z_{\alpha_{strict}} = z_{0.01} \approx 2.33$ is the 99th percentile of standard normal.

**Step 3: Safety Probability**

For a new patient $i$, the probability that their true parameter exceeds the safety bound: 

$$\mathbb{P}(\theta_i > \theta_{safe}) = \mathbb{P}\left(\frac{\theta_i - \theta_{pop}}{\sqrt{\Sigma_{between}}} > -z_{0.01}\right) = 1 - \Phi(-z_{0.01}) = \Phi(z_{0.01}) = 0.99$$

Therefore:  $\mathbb{P}(\theta_i \leq \theta_{safe}) = 0.01$

Actions are restricted to those where the population-level safety bound is satisfied: 

$$\mathcal{A}_{day1} = \{ a \in \mathcal{A} : \text{risk}(a; \theta_{safe}) \leq \delta_{safe} \}$$

Since $\theta_{safe}$ is a 99th percentile conservative bound, the probability that an action in $\mathcal{A}_{day1}$ causes a safety violation for patient $i$ is at most 0.01.

**Step 4: Relaxation**

As patient data accumulates, the posterior for $\theta_i$ concentrates around the true value. The relaxation schedule:

$$\alpha_t = \alpha_{strict} \cdot e^{-t/\tau} + \alpha_{standard} \cdot (1 - e^{-t/\tau})$$

transitions from population (conservative) to individual (less conservative) bounds while maintaining the coverage guarantee at each time point.  ∎

---

## ⚙️ Hyperparameter Settings

### Layer 1: Semantic Sensorium

| Parameter | Value |
|-----------|-------|
| Extraction model | Llama-3-8B with constrained decoding |
| Semantic entropy threshold | $\delta_{entropy} = 0.5$ |
| Number of candidate extractions | $K = 10$ |
| Sampling temperatures | $\{0.3, 0.5, 0.7, 0.9, 1.1\}$ |

### Layer 2: Digital Twin

| Parameter | Value |
|-----------|-------|
| Mechanistic model | Bergman Minimal Model |
| Neural residual | 2-layer MLP, 64 hidden units, ReLU activation |
| UKF sigma point spread | $\alpha = 0.001$, $\beta = 2$, $\kappa = 0$ |
| RBPF particles | $N = 500$ |
| Switching threshold (Shapiro-Wilk) | $p < 0.05$ |
| Switching threshold (bimodality) | $BC > 0.555$ |
| Covariance adaptation rate | $\alpha_{adapt} = 0.1$ |

### Layer 3: Causal Inference

| Parameter | Value |
|-----------|-------|
| Harmonic basis order | $K = 3$ (fundamental + 2 harmonics) |
| Randomization probability range | $p \in [0.3, 0.7]$ |
| Bridge function estimation | Kernel ridge regression, RBF kernel, $\lambda = 0.01$ |

### Layer 4: Decision Engine

| Parameter | Value |
|-----------|-------|
| Prior variance | $\sigma_0^2 = 1.0$ |
| Posterior update | Bayesian linear regression |
| Counterfactual confidence discount | $\lambda \in [0.3, 0.9]$ (Digital Twin dependent) |

### Layer 5: Safety Supervisor

| Parameter | Value |
|-----------|-------|
| Reflex threshold (hypoglycemia) | 55 mg/dL |
| Reflex threshold (hyperglycemia) | 300 mg/dL |
| STL horizon | 4 hours |
| Seldonian confidence level | $\alpha = 0.05$ (standard), $\alpha = 0.01$ (Day 1) |
| Relaxation time constant | $\tau = 14$ days |

---

## 📈 Extended Results

### Per-Patient Category Results

| Category | N | PID | Naive RL | JITAI | DT Only | AEGIS 3.0 |
|----------|---|-----|----------|-------|---------|-----------|
| Children | 10 | 58. 1 ± 5.2 | 54.3 ± 7.1 | 63.2 ± 4.5 | 66.1 ± 4.2 | **75.4 ± 3.6** |
| Adolescents | 10 | 61.8 ± 4.1 | 57.9 ± 6.2 | 66.8 ± 3.9 | 68.9 ± 3.4 | **77.8 ± 2.8** |
| Adults | 10 | 67.0 ± 3.1 | 63.8 ± 5.4 | 71.2 ± 2.9 | 73.1 ± 2.8 | **81.4 ± 2.2** |

AEGIS 3.0 shows consistent improvement across all patient categories, with largest absolute gains in the more challenging pediatric population.

### Sensitivity to Randomization Probability Range

| Probability Range | TIR (%) | Causal Bias | Effective N |
|-------------------|---------|-------------|-------------|
| [0.1, 0.9] | 77.8 ± 3.1 | 0.03 ± 0.02 | 0.18T |
| [0.2, 0.8] | 78.0 ± 2.9 | 0.04 ± 0.02 | 0.24T |
| [0.3, 0.7] | 78.2 ± 2.9 | 0.05 ± 0.03 | 0.42T |
| [0.4, 0.6] | 77.5 ± 3.2 | 0.08 ± 0.04 | 0.48T |
| [0.5, 0.5] (fixed) | 76.1 ± 3.8 | 0.12 ± 0.06 | 0.50T |

Performance is robust across reasonable randomization ranges.  Fixed 50-50 randomization reduces adaptivity but maintains causal validity.

### Long-Term Stability (24 weeks)

| Week | TIR (%) | Safety Viol.  | Adaptation Events |
|------|---------|--------------|-------------------|
| 1-4 | 72.1 ± 4.2 | 0.0 | 12. 3 ± 3.1 |
| 5-8 | 78.2 ± 2.9 | 0.0 | 4.2 ± 1.8 |
| 9-12 | 79.8 ± 2.4 | 0.0 | 2.1 ± 0.9 |
| 13-16 | 80.4 ± 2.2 | 0.0 | 1.4 ± 0.6 |
| 17-20 | 80.9 ± 2.1 | 0.0 | 1.2 ± 0.5 |
| 21-24 | 81.2 ± 2.0 | 0.0 | 1.0 ± 0.4 |

AEGIS 3.0 shows continued improvement over 24 weeks with decreasing adaptation frequency, indicating stable convergence. 

### Statistical Analysis

All comparisons use paired t-tests with Bonferroni correction for multiple comparisons.  Effect sizes reported as Cohen's d.  Confidence intervals are 95% using bootstrap resampling (1000 iterations).

| Comparison | t-statistic | p-value | Cohen's d |
|------------|-------------|---------|-----------|
| AEGIS vs.  PID | t(29) = 8.42 | p < 0.001 | 1.54 |
| AEGIS vs.  Naive RL | t(29) = 9.18 | p < 0.001 | 1.68 |
| AEGIS vs.  JITAI | t(29) = 6.21 | p < 0.001 | 1.13 |
| AEGIS vs. DT Only | t(29) = 4.87 | p < 0.001 | 0.89 |

---

## ⚠️ Limitations & Future Directions

### Current Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Proxy Validity** | Proximal causal inference requires valid negative control proxies.  Not all unmeasured confounders admit text-based proxies (e.g., genetic variants leave no narrative trace). | Automated proxy validity verification remains an open problem |
| **Simulator Fidelity** | While UVA/Padova is FDA-accepted, simulation cannot capture all real-world variability (patient behavior, sensor failures, communication delays). | Clinical validation remains necessary |
| **Computational Complexity** | RBPF with sufficient particles for multimodal tracking remains computationally demanding. Current implementation restricts RBPF to 500 particles. | Potentially limits fidelity for highly complex distributions |
| **Population Prior Quality** | Cold-start safety depends on population prior validity. For rare diseases or novel treatments without historical data, the hierarchical framework provides limited benefit. | Reduced safety guarantees for rare populations |
| **Single Disease Application** | Evaluation limited to Type 1 Diabetes. | Generalization to other chronic conditions requires validation |

### Future Directions

#### Clinical Validation

Phased approach for prospective clinical validation: 

| Phase | Description | N |
|-------|-------------|---|
| **Phase I** | Retrospective validation on existing N-of-1 trial datasets | — |
| **Phase II** | Prospective pilot with clinician-in-the-loop oversight | N=30 |
| **Phase III** | Randomized comparison against standard care | TBD |

#### Technical Extensions

- **Federated Learning**: Training population priors across institutions while preserving privacy
- **Multi-Outcome Optimization**: Extending to vector-valued outcomes with Pareto-optimal treatment selection
- **Continuous Action Spaces**: Generalizing to continuous dosing optimization (e.g., precise insulin units)
- **Formal Verification of Learning Components**: Extending formal verification to bound behavior of learned components
- **Transfer Across Conditions**: Validating across chronic conditions (hypertension, depression, chronic pain)

---

## 🔬 Ethical Considerations

### Potential Benefits

AEGIS 3.0 could improve treatment outcomes for millions of patients with chronic conditions who currently receive suboptimal population-average treatments. The framework's emphasis on individual-level optimization addresses health disparities arising from population heterogeneity—patients whose physiology differs from the "average" trial participant stand to benefit most. 

### Potential Risks

Autonomous treatment optimization systems carry inherent risks of harm from algorithmic errors.  AEGIS 3.0 mitigates these through the Simplex architecture, but no system is infallible.  Overreliance on algorithmic recommendations could erode clinical judgment.  **The system should augment, not replace, clinician decision-making.**

### Transparency and Explainability

The causal inference framework provides interpretable treatment effect estimates, enabling clinicians to understand *why* recommendations are made. The separation of mechanistic (interpretable) and neural (less interpretable) components in the Digital Twin supports this goal.

### Data Privacy

Patient narratives contain sensitive information.  The architecture processes text locally for proxy extraction without transmitting raw narratives, but privacy-preserving implementations require careful engineering.

### Equity Considerations

Population priors may encode historical biases.  If historical data underrepresents certain populations, cold-start safety may be less reliable for those patients.  Careful prior construction with diverse data sources is essential. 

---

## 📚 Comparison with Prior Work

| Capability | MOST/SMART | Standard JITAI | Digital Twin Platforms | **AEGIS 3.0** |
|------------|------------|----------------|------------------------|---------------|
| **Causal Identification** | Population g-computation | Naive regression | None (predictive only) | Proximal G-estimation with text proxies |
| **Unmeasured Confounding** | Assumed absent | Assumed absent | Assumed absent | Adjusted via negative controls |
| **State Estimation** | None | Linear mixed models | Deterministic simulation | Adaptive UKF↔RBPF switching |
| **Non-Stationarity** | Pre-specified regimes | Fixed policy | Manual recalibration | Residual-driven regime detection |
| **Safety Mechanism** | Clinician override | Soft reward penalty | Alert thresholds | Formal verification (Simplex + STL) |
| **Cold Start Safety** | Conservative dosing | Trial-and-error | Not addressed | Hierarchical Bayesian priors |
| **Exploration Strategy** | Fixed randomization | ε-greedy | None | Counterfactual Thompson Sampling |

---

## 📖 Citation

If you use AEGIS 3.0 in your research, please cite: 

```bibtex
@article{aegis3_2024,
  title={AEGIS 3.0: A Unified Architecture for Safe, Causal N-of-1 Precision Medicine},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024},
  note={Under Review}
}
```

---

## 📚 References

1. Lillie, E. O., et al. (2011). The N-of-1 clinical trial:  the ultimate strategy for individualizing medicine?  *Personalized Medicine*, 8(2), 161-173.

2. Daza, E. J.  (2018). Causal analysis of self-tracked time series data using a counterfactual framework for N-of-1 trials. *Methods of Information in Medicine*, 57(S01), e10-e21.

3. Klasnja, P., et al. (2015). Microrandomized trials: An experimental design for developing just-in-time adaptive interventions. *Health Psychology*, 34(S), 1220-1228.

4. Nahum-Shani, I., et al. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health:  Key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine*, 52(6), 446-462.

5. Björnsson, B., et al. (2020). Digital twins to personalize medicine.  *Genome Medicine*, 12(1), 1-4.

6. Man, C. D., et al. (2014). The UVA/PADOVA type 1 diabetes simulator:  new features.  *Journal of Diabetes Science and Technology*, 8(1), 26-34.

7. Rackauckas, C., et al. (2020). Universal Differential Equations for Scientific Machine Learning. *arXiv:2001.04385*.

8. Lal, A., et al. (2024). Causal artificial intelligence and digital twins are transforming drug discovery.  *Nature*, d43747-024-00077-9.

9. Gottesman, O., et al. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine*, 25(1), 16-18.

10. Altman, E. (1999). *Constrained Markov Decision Processes*. CRC Press.

11. Thomas, P. S., et al. (2019). Preventing undesirable behavior of intelligent machines. *Science*, 366(6468), 999-1004.

12. Sha, L., et al. (2001). Using simplicity to control complexity.  *IEEE Software*, 18(4), 20-28.

13. Vaskov, A., et al. (2024). Do No Harm: A Counterfactual Approach to Safe Reinforcement Learning. *Proceedings of Machine Learning Research*, 242. 

14. Robins, J. M.  (1994). Correcting for non-compliance in randomized trials using structural nested mean models. *Communications in Statistics-Theory and Methods*, 23(8), 2379-2412.

15. Robins, J. M., Hernán, M.  A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*, 11(5), 550-560.

16. Tchetgen Tchetgen, E., et al. (2024). An Introduction to Proximal Causal Inference. *Statistical Science*, 39(3).

17. Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993.

18. Ying, A., Miao, W., & Tchetgen Tchetgen, E. J. (2023). Proximal causal inference for marginal structural models. *Biometrika*, asad015.

19. Veitch, V., Sridhar, D., & Blei, D. (2020). Adapting text embeddings for causal inference. *Conference on Uncertainty in Artificial Intelligence*, 919-928.

20. Kovatchev, B.  P., et al. (2009). In silico preclinical trials:  a proof of concept in closed-loop control of type 1 diabetes. *Journal of Diabetes Science and Technology*, 3(1), 44-55.

21. Särkkä, S.  (2013). *Bayesian Filtering and Smoothing*. Cambridge University Press. 

22. Doucet, A., De Freitas, N., & Gordon, N. (2001). An introduction to sequential Monte Carlo methods. *Sequential Monte Carlo Methods in Practice*, 3-14.

23. Greenewald, K., et al. (2017). Action centered contextual bandits. *Advances in Neural Information Processing Systems*, 30.

24. Howard, S. R., et al. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences.  *Annals of Statistics*, 49(2), 1055-1080.

25. Maler, O., & Nickovic, D. (2004). Monitoring temporal properties of continuous signals.  *Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems*, 152-166.

---

## 📋 Target Publication Venues

### Tier 1: Primary Targets

| Venue | Type | Impact Factor | Fit |
|-------|------|---------------|-----|
| **IEEE Journal of Biomedical and Health Informatics (JBHI)** | Journal | 7.7 | ★★★★★ |
| **npj Digital Medicine** | Journal | 15.2 | ★★★★★ |
| **Nature Machine Intelligence** | Journal | 25.9 | ★★★★☆ |

### Tier 2: Strong Alternatives

| Venue | Type | Impact Factor | Fit |
|-------|------|---------------|-----|
| **JAMIA** | Journal | 7.9 | ★★★★☆ |
| **Journal of Machine Learning Research (JMLR)** | Journal | 6.0 | ★★★★☆ |
| **Artificial Intelligence in Medicine** | Journal | 7.5 | ★★★★☆ |

### Tier 3: Conference Strategy

| Venue | Type | Fit |
|-------|------|-----|
| **MLHC 2025** | Conference | ★★★★★ |
| **CHIL 2025** | Conference | ★★★★★ |
| **NeurIPS ML4H Workshop** | Workshop | ★★★★☆ |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

<div align="center">

**AEGIS 3.0** — *Advancing the science of individualized medicine through causal inference, safe learning, and formal verification.*

Made with ❤️ for precision medicine research

</div>
