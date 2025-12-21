
# AEGIS 3.0: A Unified Architecture for Safe, Causal N-of-1 Precision Medicine

## Abstract

The promise of precision medicine—delivering the right treatment to the right patient at the right time—remains unrealized due to a fundamental epistemological gap between population-derived evidence and individual therapeutic response. The Average Treatment Effect (ATE), the cornerstone of Evidence-Based Medicine, rests on an ergodicity assumption that demonstrably fails in complex biological systems characterized by non-stationarity, path dependence, and feedback dynamics. This paper presents **AEGIS 3.0** (Adaptive Engineering for Generalized Individualized Safety), a five-layer unified architecture that synthesizes advances in causal inference, Bayesian state estimation, and formal verification to enable provably safe, causally valid treatment optimization for the individual patient. 

AEGIS 3.0 introduces four principal algorithmic innovations: (1) **Proximal G-Estimation with Text-Derived Negative Controls**, enabling causal identification under unmeasured confounding by leveraging semantic features from patient narratives; (2) **Adaptive Hybrid State Estimation** via automatic switching between Adaptive Constrained Unscented Kalman Filters (AC-UKF) and Rao-Blackwellized Particle Filters (RBPF) based on detected distributional regime; (3) **Counterfactual Thompson Sampling (CTS)**, a novel bandit algorithm that maintains exploration efficiency under hard safety constraints through Digital Twin-imputed posterior updates; and (4) **Hierarchical Cold-Start Seldonian Constraints** with population-derived Bayesian priors for initial safety guarantees without patient-specific data.

We provide formal identification theorems, regret bounds under safety constraints, and comprehensive in-silico validation using the FDA-accepted UVA/Padova Type 1 Diabetes simulator. Results demonstrate significant improvements in time-in-range (78. 2% vs.  62.3% for standard control) with zero safety violations across 30 virtual patients over 8-week simulated trials.  This work establishes methodological foundations for causally valid, safe adaptive treatment optimization in N-of-1 settings.

**Keywords**:  N-of-1 Trials, Causal Inference, Digital Twins, Safe Reinforcement Learning, Precision Medicine, Micro-Randomized Trials, Proximal Causal Inference, Thompson Sampling

---

## 1. Introduction

### 1.1 The Precision Medicine Challenge

For five decades, the Randomized Controlled Trial (RCT) has served as the epistemological gold standard for therapeutic evidence. The statistical validity of applying population-derived conclusions to individual patients rests on an implicit assumption borrowed from statistical mechanics: **ergodicity**—the equivalence of ensemble averages (across patients at one time) and time averages (within one patient across time). Formally: 

$$\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^{N} Y_i(t) \stackrel{? }{=} \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} Y_i(t)$$

In complex adaptive systems—including human physiology—this equality **demonstrably fails**. Biological systems exhibit hysteresis (history-dependent responses), non-stationarity (time-varying dynamics), and bifurcations (qualitative regime changes). A medication producing a positive Average Treatment Effect (ATE) may be inert, suboptimal, or harmful for a specific individual due to idiosyncratic genetic, environmental, or physiological boundary conditions.

This represents not merely statistical noise to be averaged away, but a **structural inadequacy** of population statistics to characterize individual response.  The transition from population-level inference to individual-level optimization—from "What works on average?" to "What works for *this* patient *now*?"—constitutes the defining computational challenge of twenty-first century medicine.

### 1.2 The Small Data Paradox

The N-of-1 trial, wherein a single patient serves as their own control across multiple treatment periods, offers a principled solution to the ergodicity problem. However, this design introduces a complementary challenge: the **Small Data Paradox**.  Modern machine learning achieves its power through massive datasets where the Law of Large Numbers suppresses variance.  In N-of-1 trials, we possess perhaps T=100 observations for a single individual—insufficient for data-hungry deep learning yet exhibiting complex temporal dependencies that violate classical statistical assumptions.

Previous attempts to address this paradox have produced instructive failures: 

| Architecture | Approach | Failure Mode |
|--------------|----------|--------------|
| **VACA-type** | Predictive deep learning (LSTM/RNN) | Confounding by indication; conflated correlation with causation |
| **Discovery-based** | Data-driven causal discovery | Structural instability; hallucinated causal links from sparse data |
| **Standard RL** | Reinforcement learning | Unsafe exploration; sample inefficiency in short trials |

These failures share a common root: attempting to learn complex dynamics *de novo* from radically insufficient data, while ignoring both the rich prior knowledge encoded in physiological science and the safety imperatives of medical intervention.

### 1.3 The Case for Unified Architecture

Existing approaches address individual challenges—causal inference, state estimation, safe learning, or cold-start—in isolation. However, N-of-1 precision medicine requires their **simultaneous** resolution. A system with perfect causal inference but unsafe exploration will harm patients.  A perfectly safe system that cannot identify causal effects will deliver suboptimal treatment. A system lacking cold-start capabilities cannot be deployed to new patients.

AEGIS 3.0 resolves the Small Data Paradox through a **Grey-Box** architecture that embeds mechanistic physiological priors while learning patient-specific deviations. It addresses the safety imperative through **formal verification** that decouples learning from constraint enforcement.  And it achieves causal validity through **design-based identification** augmented by novel methods for unmeasured confounding adjustment.

### 1.4 Contributions

This paper makes three categories of contributions:

**C1: Architectural Integration (Primary)**
We present AEGIS 3.0, the first unified architecture that jointly addresses causal identification, state estimation, safe exploration, and cold-start safety for N-of-1 trials. While individual components draw on existing techniques, their integration is novel and non-trivial—we identify key interface requirements and resolve tensions between competing objectives (e.g., exploration vs. safety). We articulate four design principles that guide architectural decisions and enable principled extension.

**C2: Algorithmic Innovations (Secondary)**
We introduce four novel algorithmic contributions:
1. **Proximal G-Estimation with Text-Derived Negative Controls**: First application of proximal causal inference to N-of-1 trials using semantic features from patient narratives as treatment/outcome confounding proxies
2. **Counterfactual Thompson Sampling (CTS)**: Bandit algorithm maintaining posterior updates for safety-blocked actions through model-imputed counterfactual outcomes with confidence-weighted likelihood
3. **Hierarchical Cold-Start Seldonian Constraints**: Framework for transferring population-level safety posteriors to individual patients, enabling probabilistic safety guarantees without patient-specific adverse event data
4. **Adaptive Hybrid State Estimation**: Principled criterion for automatic selection between AC-UKF and RBPF based on detected distributional regime

**C3: Validation Framework (Tertiary)**
We develop a comprehensive in-silico validation protocol using the FDA-accepted UVA/Padova Type 1 Diabetes simulator, enabling rigorous evaluation of N-of-1 systems with clinically meaningful endpoints.

### 1.5 Paper Organization

Section 2 reviews background and related work.  Section 3 formalizes the N-of-1 causal control problem.  Section 4 presents design principles and architectural overview. Section 5 details layer specifications.  Section 6 provides theoretical analysis. Section 7 presents experimental evaluation. Section 8 discusses limitations and future directions. Section 9 concludes. 

---

## 2. Background and Related Work

### 2.1 N-of-1 Trials and Micro-Randomized Trials

N-of-1 trials represent a rigorous design for individual-level treatment effect estimation, with origins in behavioral psychology and adoption in chronic disease management [1]. The patient serves as their own control across multiple crossover periods, enabling within-subject causal inference. Recent formalization has established conditions for valid causal effect estimation, including handling of carryover effects and time-varying confounding [2].

**Micro-Randomized Trials (MRTs)** extend this paradigm to intensive longitudinal settings with hundreds of decision points [3]. At each decision point, treatment is randomized with known probability conditional on context, enabling causal identification of time-varying effects. MRTs form the foundation for **Just-In-Time Adaptive Interventions (JITAIs)** in mobile health [4].

### 2.2 Digital Twins in Healthcare

Digital Twins—computational models that mirror individual patient physiology—have emerged as enabling technology for precision medicine [5]. In diabetes management, models like the Bergman Minimal Model and UVA/Padova simulator provide mechanistic foundations [6]. Recent work combines mechanistic models with data-driven components through **Universal Differential Equations (UDEs)**, learning residual dynamics while preserving interpretability [7].

However, existing Digital Twin approaches typically focus on prediction without causal identification, conflating correlation with causation in treatment optimization [8]. 

### 2.3 Safe Reinforcement Learning

Safe exploration remains a fundamental challenge in sequential decision-making for healthcare [9].  Approaches include: 
- **Constrained MDPs**: Optimize reward subject to constraint satisfaction [10]
- **Seldonian Algorithms**: Provide high-confidence bounds on safety-relevant quantities [11]
- **Simplex Architecture**: Decouple verified safety from unverified learning [12]

Recent work on counterfactual approaches to safe RL [13] distinguishes inevitable constraint violations from agent-caused harm—conceptually related but distinct from our Counterfactual Thompson Sampling. 

### 2.4 Causal Inference for Adaptive Interventions

Causal inference methods for longitudinal data include **G-estimation** for structural nested mean models [14], **marginal structural models** with inverse probability weighting [15], and recent developments in **proximal causal inference** for unmeasured confounding [16, 17]. 

Proximal causal inference leverages negative control variables—proxies for unmeasured confounders—to achieve identification.  Recent work has extended these methods to time-series settings [18] and explored text-derived proxies from clinical notes [19].  Our contribution extends this line to N-of-1 trials with patient-generated narratives. 

### 2.5 Positioning:  How AEGIS 3.0 Differs

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

## 3. Problem Formalization

### 3.1 The N-of-1 Causal Control Problem

**Definition 3.1 (N-of-1 Causal Control)**: Let a single patient be characterized by: 

- **Observable State**: $S_t \in \mathcal{S} \subseteq \mathbb{R}^p$, a vector of clinical variables at time $t$
- **Hidden Physiological State**: $X_t \in \mathcal{X} \subseteq \mathbb{R}^n$, latent variables governing dynamics
- **Observations**: $Y_t \in \mathcal{Y}$, noisy measurements of outcomes
- **Treatment Actions**: $A_t \in \mathcal{A}$, the intervention space
- **Patient Narrative**: $\mathcal{T}_t \in \Sigma^*$, unstructured text (diaries, messages)
- **History**: $H_t = \{S_{1:t}, A_{1:t-1}, Y_{1:t}, \mathcal{T}_{1:t}\}$, all information to time $t$

The objective is to find a policy $\pi:  H_t \mapsto A_t$ that minimizes **Individual Regret**:

$$\mathcal{R}(\pi, T) = \sum_{t=1}^{T} \left[ Y_t^{*(a_t^*)} - \mathbb{E}[Y_t \mid \text{do}(A_t = \pi(H_t)), H_t] \right]$$

where $a_t^* = \arg\max_a \mathbb{E}[Y_t \mid \text{do}(A_t = a), H_t]$ is the **optimal Individual Treatment Effect (ITE)** and $Y_t^{*(a)}$ denotes the potential outcome under intervention $a$.

### 3.2 Safety Constraints

**Definition 3.2 (Safety Constraints)**: The policy must satisfy:

- **Hard Constraints** (Signal Temporal Logic): $\Box_{[0,T]}(\phi_{safety})$ where $\phi_{safety}$ encodes inviolable physiological boundaries (e.g., glucose > 70 mg/dL)

- **Probabilistic Constraints** (Seldonian): $\mathbb{P}(g(\theta) > 0) \leq \alpha$ for safety-relevant functions $g$ (e.g., probability of adverse event exceeding threshold)

### 3.3 Identification Challenges

Causal identification of the ITE requires the **Sequential Ignorability** assumption:

$$Y_{t+1}^{\bar{a}} \perp\!\!\!\perp A_t \mid H_t \quad \forall t, \bar{a}$$

This assumption—that treatment assignment is independent of potential outcomes given observed history—is **routinely violated** in N-of-1 trials due to: 

1. **Unmeasured Time-Varying Confounding**:  Factors like stress, sleep quality, or environmental exposures affect both treatment decisions and outcomes but may not be captured in structured data. 

2. **Circadian Confounding**: Time-of-day systematically influences both patient availability for treatment and physiological response, creating spurious treatment-outcome associations.

3. **Feedback Dynamics**: Past outcomes influence future treatment decisions through adaptive behavior, creating complex causal chains. 

AEGIS 3.0 addresses each challenge through architectural innovations detailed in subsequent sections.

---

## 4. The AEGIS 3.0 Architecture

### 4.1 Design Principles

The AEGIS 3.0 architecture is guided by four principles derived from the unique challenges of N-of-1 precision medicine:

**Principle 1: Grey-Box Integration**
Pure data-driven approaches fail with small N-of-1 data (insufficient samples for complex function approximation). Pure mechanistic models cannot capture individual variation (parameters derived from population averages). AEGIS integrates both:  mechanistic priors constrain the hypothesis space to physiologically plausible trajectories while learned components capture patient-specific deviations.

*Architectural implication*: Layer 2 implements Universal Differential Equations combining fixed physiological models with neural residuals. 

**Principle 2: Separation of Learning and Safety**
Learning systems must explore to improve; safety systems must be conservative.  Coupling these creates tension that either compromises learning (over-conservative) or safety (under-conservative). AEGIS decouples them through the Simplex architecture:  the learning system operates freely while an independent verified supervisor enforces constraints.

*Architectural implication*:  Layers 1-4 learn and optimize; Layer 5 independently verifies and can override any decision.

**Principle 3: Design-Based Causal Identification**
Observational inference from N-of-1 data is confounded—patients modify behavior based on symptoms, creating treatment-outcome associations that are not causal. AEGIS embeds randomization (MRTs) into the decision process, enabling causal identification by design rather than assumption.

*Architectural implication*: Layer 4 implements micro-randomization with known probabilities; Layer 3 exploits these for unbiased causal estimation.

**Principle 4: Hierarchical Information Transfer**
Cold-start is inevitable for new patients—no patient-specific data exists on Day 1. Purely patient-specific methods require dangerous exploration periods. AEGIS transfers population knowledge hierarchically, constraining initial behavior while permitting personalization as data accumulates.

*Architectural implication*: All layers maintain hierarchical priors that relax from population to individual as evidence accumulates.

### 4.2 Architecture Overview

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
│    │    RAW DATA:  Wearables │ CGM │ PRO Surveys │ Patient Diaries       │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Inter-Layer Communication Protocol

Information flows bidirectionally through the architecture: 

**Upward Flow (Inference)**:
- Layer 1 → Layer 2: Structured observations $(S_t, Y_t)$ and proxy variables $(Z_t, W_t)$
- Layer 2 → Layer 3: State estimates $\hat{x}_t$ with uncertainty $P_t$
- Layer 3 → Layer 4: Treatment effect estimates $\hat{\tau}(S_t)$ with confidence bounds
- Layer 4 → Layer 5: Proposed action $A_t^{proposed}$ for safety verification

**Downward Flow (Control)**:
- Layer 5 → Layer 4: Safety-certified action $A_t^{safe}$ or blocking signal
- Layer 4 → Layer 3: Randomization probability $p_t(S_t)$ for causal inference
- Layer 3 → Layer 2: Causal constraints for counterfactual simulation
- Layer 2 → Layer 1: Expected observations for anomaly detection

---

## 5. Layer Specifications

### 5.1 Layer 1: Semantic Sensorium

#### 5.1.1 Problem Statement

N-of-1 digital trials generate heterogeneous data streams:  continuous sensor measurements, periodic surveys, and unstructured patient narratives. The data layer must accomplish three objectives:

1. **Semantic Standardization**: Map diverse inputs to a consistent clinical ontology
2. **Uncertainty Quantification**: Detect and flag unreliable extractions
3. **Causal Proxy Identification**: Extract variables suitable for confounding adjustment

#### 5.1.2 Ontology-Constrained Extraction

AEGIS 3.0 enforces semantic consistency through **constrained generation**.  Rather than extracting free-form text, the extraction module maps patient narratives to SNOMED-CT concept identifiers through grammatically constrained decoding.  This ensures that semantically equivalent expressions ("drowsy," "sleepy," "tired," "zonked out") map to identical nodes in the causal graph, preventing artificial sparsity. 

**Specification 5.1 (Extraction Output Schema)**:
```
Observation := {
    concept_id:   SNOMED-CT Identifier,
    value:       Numeric ∪ Categorical,
    unit:        UCUM Standard Unit,
    timestamp:   ISO-8601 with mandatory timezone,
    confidence:   [0, 1],
    semantic_entropy: [0, ∞)
}
```

#### 5.1.3 Semantic Entropy Thresholding

Standard confidence scores fail to capture *semantic* uncertainty. A model may assign 95% probability to an extraction while being fundamentally uncertain about its meaning. AEGIS 3.0 implements **Semantic Entropy** quantification: 

1. Generate K candidate extractions with varying sampling temperatures
2. Embed candidates in SNOMED-CT semantic space
3. Cluster candidates by semantic equivalence (same concept ID)
4. Compute entropy over cluster distribution: 

$$H_{sem}(\mathcal{T}_t) = -\sum_{c \in \mathcal{C}} p(c) \log p(c)$$

where $p(c)$ is the proportion of candidates falling in semantic cluster $c$.

**Decision Rule**:  Trigger Human-in-the-Loop (HITL) review when $H_{sem} > \delta_{entropy}$, indicating semantically distinct interpretations with non-trivial probability.

#### 5.1.4 Causal Role Classification for Proximal Inference (Novel Contribution)

A principal innovation of AEGIS 3.0 is leveraging patient narratives as sources of **negative control proxies** for unmeasured confounding adjustment. This requires classifying extracted semantic features by their causal role. 

**Definition 5.1 (Treatment-Confounder Proxy)**: A variable $Z_t$ extracted from text serves as a valid treatment-confounder proxy if: 
- $Z_t \perp\!\!\!\perp Y_t \mid U_t, S_t$ (no direct effect on outcome)
- $Z_t \not\perp\!\!\!\perp U_t \mid S_t$ (associated with unmeasured confounder)
- $Z_t \perp\!\!\!\perp A_t \mid U_t, S_t$ (not caused by treatment)

**Definition 5.2 (Outcome-Confounder Proxy)**: A variable $W_t$ serves as a valid outcome-confounder proxy if:
- $W_t \perp\!\!\!\perp A_t \mid U_t, S_t$ (not caused by treatment)
- $W_t \not\perp\!\!\!\perp U_t \mid S_t$ (associated with unmeasured confounder)

**Example**:  Consider unmeasured psychological stress ($U_t$) affecting both medication adherence ($A_t$) and symptom severity ($Y_t$). Patient diary mentions of "work deadline" ($Z_t$) may serve as treatment-proxy (stress causes deadline mention; deadline doesn't directly affect symptoms). Mentions of "couldn't sleep" ($W_t$) may serve as outcome-proxy (stress causes poor sleep; poor sleep predicts symptoms but isn't caused by today's treatment).

The Semantic Sensorium applies rule-based classification augmented by temporal precedence analysis to assign proxy roles.

### 5.2 Layer 2: Adaptive Digital Twin

#### 5.2.1 Universal Differential Equations

The Digital Twin maintains a dynamic model of patient physiology through **Universal Differential Equations (UDEs)**:

$$\frac{dx}{dt} = f_{mech}(x, u; \theta_{fixed}) + f_{NN}(x, u; \theta_{learned})$$

where:
- $f_{mech}$ encodes established physiological mechanisms (e.g., insulin-glucose dynamics via the Bergman Minimal Model)
- $f_{NN}$ is a neural network learning patient-specific deviations from textbook physiology
- $\theta_{fixed}$ are literature-derived parameters
- $\theta_{learned}$ are personalized parameters estimated from patient data

This architecture resolves the Small Data Paradox:  the mechanistic prior constrains the hypothesis space to physiologically plausible trajectories, while the neural residual captures individual variation.

#### 5.2.2 Adaptive Constrained UKF (AC-UKF)

For unimodal state distributions, AEGIS 3.0 implements the **Adaptive Constrained UKF** with two innovations:

**Innovation-Based Covariance Adaptation**: The filter monitors measurement residuals $\epsilon_k = y_k - h(\hat{x}_k^-)$. If empirical residual variance exceeds theoretical prediction, process noise covariance $Q_k$ is inflated:

$$Q_{k+1} = Q_k + \alpha K_k \left( \epsilon_k \epsilon_k^T - S_k \right) K_k^T$$

where $S_k$ is the predicted residual covariance and $K_k$ is the Kalman gain.

**Constraint Projection**: Before propagating sigma points through the ODE, a projection operator enforces physiological constraints:

$$\mathcal{X}_{sigma}^{proj} = \Pi_{\mathcal{C}}(\mathcal{X}_{sigma})$$

This prevents numerical instabilities from unphysical states. 

#### 5.2.3 Rao-Blackwellized Particle Filter (RBPF)

When state distributions become multimodal—during regime transitions, disease exacerbations, or bifurcation events—Gaussian approximations fail categorically. AEGIS 3.0 employs **RBPF** for such regimes. 

RBPF exploits conditional linearity:  partition states into $x = [x_{lin}, x_{nl}]$ where linear dynamics govern $x_{lin}$ conditional on $x_{nl}$.  The posterior factorizes: 

$$p(x_{lin}, x_{nl} \mid y_{1:t}) = p(x_{lin} \mid x_{nl}, y_{1:t}) \cdot p(x_{nl} \mid y_{1:t})$$

The linear component admits closed-form Kalman updates; only the nonlinear component requires particle approximation.

#### 5.2.4 Automatic Filter Selection (Novel Contribution)

AEGIS 3.0 implements automatic switching based on distribution diagnostics:

**Switching Criterion**: At each timestep, evaluate:
1. **Normality Test**: Shapiro-Wilk statistic on recent residuals
2. **Bimodality Coefficient**: $BC = \frac{skewness^2 + 1}{kurtosis}$

**Decision Rule**:
- If Shapiro-Wilk $p < 0.05$ OR $BC > 0.555$: Deploy RBPF (non-Gaussian/multimodal detected)
- Otherwise: Deploy AC-UKF (Gaussian adequate)
- If RBPF effective sample size drops below threshold:  Trigger resampling

### 5.3 Layer 3: Causal Inference Engine

#### 5.3.1 Micro-Randomized Trial Design

At each decision point $k$, treatment $A_k$ is randomized with probability $p_k(S_k)$ conditional on observed context.  This design maximizes effective sample size while maintaining causal identification through known randomization probabilities.

**Positivity Constraint**: $\epsilon < p_k(S_k) < 1 - \epsilon$ for all contexts, ensuring all treatment-context combinations remain possible.

#### 5.3.2 Harmonic Time-Varying G-Estimation

Standard G-estimation assumes time-invariant treatment effects. AEGIS 3.0 implements **Harmonic G-Estimation** with time-varying effects:

**Baseline Model** (Fourier decomposition):
$$\mu(t; \beta) = \beta_0 + \sum_{k=1}^{K} \left[ \beta_{ck} \cos\left(\frac{2\pi k t}{24}\right) + \beta_{sk} \sin\left(\frac{2\pi k t}{24}\right) \right]$$

**Treatment Effect Model** (time-varying):
$$\tau(t; \psi) = \psi_0 + \sum_{k=1}^{K} \left[ \psi_{ck} \cos\left(\frac{2\pi k t}{24}\right) + \psi_{sk} \sin\left(\frac{2\pi k t}{24}\right) \right]$$

**Estimating Equation**:
$$\sum_{t=1}^{T} \left[ Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi) A_t \right] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t) = 0$$

This formulation allows treatment effects to **vary by time of day** while orthogonalizing against circadian baseline variation.

#### 5.3.3 Double Robustness Property

**Theorem 5.1 (Double Robustness)**: Under positivity and consistency assumptions, the Harmonic G-estimator $\hat{\psi}$ converges in probability to the true effect $\psi^*$ if either:
1. $\hat{\mu}(S_t) = \mathbb{E}[Y_{t+1} \mid S_t, A_t=0]$ (outcome model correctly specified), OR
2. $p_t(S_t) = \mathbb{P}(A_t=1 \mid S_t)$ (propensity model correctly specified)

Since randomization probabilities are determined algorithmically in MRTs, condition (2) is satisfied by construction.

#### 5.3.4 Proximal G-Estimation for Unmeasured Confounding (Novel Contribution)

When unmeasured confounders $U_t$ violate sequential ignorability, standard G-estimation produces biased effect estimates. AEGIS 3.0 integrates **Proximal Causal Inference** using text-derived negative controls. 

**Assumption 5.1 (Proxy Completeness)**: The treatment-confounder proxy $Z_t$ and outcome-confounder proxy $W_t$ satisfy:
$$\text{span}\{\mathbb{E}[h(W) \mid Z, S]\} = L^2(U \mid S)$$

Under this richness condition, a **Bridge Function** $h^*(W_t)$ exists such that adjustment recovers the causal effect despite $U_t$ being unobserved.

**Augmented Estimating Equation**:
$$\sum_{t=1}^{T} \left[ Y_{t+1} - \hat{\mu}(S_t) - \tau(t; \psi) A_t - h^*(W_t) \right] \cdot (A_t - p_t(S_t)) \cdot \mathbf{h}(t) = 0$$

**Theorem 5.2 (Proximal Identification)**: Under Assumption 5.1 and standard regularity conditions, the proximal G-estimator identifies the causal effect $\psi^*$ even when $U_t \not\in H_t$.

#### 5.3.5 Anytime-Valid Inference

Adaptive trials require **continuous monitoring** without inflating Type-I error.  AEGIS 3.0 employs **Martingale Confidence Sequences** that maintain coverage guarantees at arbitrary stopping times.

**Definition 5.3 (Confidence Sequence)**: A sequence of confidence sets $\{CS_t\}_{t=1}^{\infty}$ is $(1-\alpha)$-valid if:
$$\mathbb{P}\left( \psi^* \in CS_t \text{ for all } t \geq 1 \right) \geq 1 - \alpha$$

### 5.4 Layer 4: Decision Engine

#### 5.4.1 Action-Centered Contextual Bandits

Standard reinforcement learning attempts to learn the total reward function $Q(S, A)$. In N-of-1 trials, reward variance is dominated by baseline health fluctuations unrelated to treatment. AEGIS 3.0 employs **Action-Centered Bandits** that decompose reward: 

$$R_t = f(S_t) + A_t \cdot \tau(S_t) + \epsilon_t$$

The bandit learns *only* $\tau(S_t)$—the treatment effect—treating $f(S_t)$ as noise to be subtracted.  This **variance reduction** accelerates learning by orders of magnitude.

**Theorem 5.3 (Regret Bound)**: The Action-Centered Bandit achieves regret: 
$$\mathcal{R}(T) = \tilde{O}(d_{\tau} \sqrt{T})$$
where $d_{\tau}$ is the dimension of treatment effect parameters. 

#### 5.4.2 Counterfactual Thompson Sampling (Novel Contribution)

Standard constrained bandits create a **pathology**:  if the optimal action lies near the safety boundary, it may be repeatedly blocked. The posterior for this action never updates—**posterior collapse**—leaving the system uncertain about potentially excellent treatments indefinitely.

AEGIS 3.0 introduces **Counterfactual Thompson Sampling (CTS)**:

**Algorithm 5.1 (Counterfactual Thompson Sampling)**: 

```
Input: Posterior P(θ | H_t), safety evaluator S, Digital Twin D

1. SAMPLE:  Draw θ̃ ~ P(θ | H_t)

2. OPTIMIZE:  Compute unconstrained optimum a* = argmax_a E[R | S_t, a, θ̃]

3. SAFETY CHECK: Query safety supervisor for a*
   - If S(a*, S_t) = SAFE: Execute A_t = a*
   - If S(a*, S_t) = UNSAFE: Proceed to Step 4

4. COUNTERFACTUAL UPDATE (for blocked action a*):
   - Impute counterfactual outcome:  Ŷ_{a*} = D. predict(S_t, a*)
   - Compute imputation confidence: λ = D.confidence(S_t, a*)
   - Update posterior with discounted likelihood:
     P(θ | H_{t+1}) ∝ P(Ŷ_{a*} | θ, S_t, a*)^λ · P(θ | H_t)

5. SAFE SELECTION: Execute A_t = argmax_{a ∈ A_safe} E[R | S_t, a, θ̃]
```

**Key Innovation**: Step 4 updates the posterior for the blocked action using Digital Twin predictions.  The discount factor $\lambda \in (0, 1)$ reflects imputation uncertainty—high confidence yields stronger updates; low confidence yields weak updates.

**Theorem 5.4 (CTS Regret Bound)**: Under bounded rewards, accurate safety constraints, and bounded imputation error, CTS achieves: 
$$\mathcal{R}(T) \leq \tilde{O}(d_{\tau} \sqrt{T \log T}) + O(B_T \cdot \Delta_{max} \cdot (1-\lambda))$$
where $B_T$ is the number of blocking events and $\Delta_{max}$ is the maximum suboptimality gap.

### 5.5 Layer 5: Simplex Safety Supervisor

#### 5.5.1 Three-Tier Safety Hierarchy

AEGIS 3.0 implements three safety tiers with strict priority ordering:

**Tier 1:  Reflex Controller (Highest Priority)**
- **Mechanism**: Model-free threshold logic operating directly on sensor measurements
- **Examples**: "If glucose < 55 mg/dL, halt all insulin recommendations"
- **Rationale**: Cannot be fooled by Digital Twin errors; operates on raw reality

**Tier 2: STL Monitor (Signal Temporal Logic)**
- **Mechanism**: Formal verification of predicted trajectories against temporal specifications
- **Specifications**:  Expressed in STL, e.g., $\Box_{[0,T]}(G > 70) \wedge \Box_{[0,T]}(G < 250)$
- **Computation**: Reachability analysis using conservative physiological bounds

**Tier 3: Seldonian Constraints (Probabilistic)**
- **Mechanism**: High-confidence bounds on safety-relevant outcome probabilities
- **Specification**: $\mathbb{P}(g(\theta) > 0) \leq \alpha$ for constraint function $g$

**Conflict Resolution**: When tiers disagree, higher-priority tier prevails. 

#### 5.5.2 Breaking the Circularity Problem

Previous approaches suffered from **safety circularity**: the STL monitor relied on Digital Twin predictions; if the Twin diverged, safety checks became meaningless. 

AEGIS 3.0 breaks this circularity through **Reachability Analysis** using population-derived worst-case bounds independent of the patient-specific Digital Twin: 

**Definition 5.4 (Conservative Physiological Bounds)**: For physiological variable $x$, define:
- Maximum rate of change: $|\dot{x}| \leq \dot{x}_{max}$ (from population studies)
- Action delay bounds: $t_{onset} \in [t_{min}, t_{max}]$, $t_{peak} \in [t_{min}', t_{max}']$
- Physiological limits: $x \in [x_{min}, x_{max}]$

**Reachability Set**:  For current state $x_t$ and proposed action $a_t$, compute worst-case future states:
$$\mathcal{R}_{t+\Delta}(x_t, a_t) = \{ x' : \exists \text{ trajectory from } x_t \text{ under } a_t \text{ respecting bounds} \}$$

**Safety Decision**:
$$A_{final} = \begin{cases}
A_{complex} & \text{if } \mathcal{R}_{t+\Delta} \cap \mathcal{X}_{unsafe} = \emptyset \\
A_{reflex} & \text{otherwise}
\end{cases}$$

#### 5.5.3 Cold Start Safety via Hierarchical Priors (Novel Contribution)

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

## 6. Theoretical Analysis

### 6.1 Identification Theorems

**Theorem 6.1 (Harmonic G-Estimation Identification)**: Under the assumptions of: 
1. **Consistency**: $Y_t = Y_t^{a}$ when $A_t = a$
2. **Positivity**: $\epsilon < p_t(S_t) < 1 - \epsilon$ for all $t, S_t$
3. **Sequential Ignorability**: $Y_{t+1}^{\bar{a}} \perp\!\!\!\perp A_t \mid H_t$

The Harmonic G-estimator identifies the time-varying causal effect: 
$$\tau(t; \psi^*) = \mathbb{E}[Y_{t+1}^{1} - Y_{t+1}^{0} \mid S_t, t]$$

*Proof sketch*: The estimating equation is unbiased by construction of the MRT. The Fourier basis spans the space of smooth periodic functions, capturing circadian variation. Full proof in Appendix A.

**Theorem 6.2 (Proximal Identification)**: When sequential ignorability fails due to unmeasured confounder $U_t$, but valid proxies $(Z_t, W_t)$ exist satisfying Assumption 5.1, the Proximal G-estimator identifies the causal effect $\psi^*$ even when $U_t \not\in H_t$.

*Proof sketch*: Under completeness, the bridge function $h^*(W_t) = \mathbb{E}[U_t \mid W_t, S_t]$ spans the confounding adjustment needed. The augmented estimating equation removes the bias term.  Full proof in Appendix B. 

**Theorem 6.3 (Double Robustness)**: The combined estimator is consistent if either:
1. The Digital Twin correctly specifies $\mathbb{E}[Y_{t+1} \mid S_t, A_t = 0]$, OR
2. The randomization probabilities $p_t(S_t)$ are correctly specified (true by design in MRTs)

### 6.2 Regret Analysis

**Theorem 6.4 (Safe Exploration Regret)**: Under the AEGIS 3.0 architecture with CTS, total regret satisfies: 

$$\mathcal{R}(T) \leq \underbrace{O(d_{\tau} \sqrt{T \log T})}_{\text{Learning regret}} + \underbrace{O(B_T \cdot \Delta_{max} \cdot (1-\lambda))}_{\text{Safety blocking regret}}$$

where: 
- $d_{\tau}$ = dimension of treatment effect parameters
- $B_T$ = number of safety-blocked decisions
- $\Delta_{max}$ = maximum suboptimality of safe alternatives
- $\lambda$ = average imputation confidence

*Proof sketch*: The first term follows from standard Thompson Sampling analysis. The second term bounds the cost of blocking.  Counterfactual updates with confidence $\lambda$ reduce effective blocking count. As Digital Twin improves ($\lambda \to 1$), blocking regret vanishes. Full proof in Appendix C.

### 6.3 Safety Guarantees

**Theorem 6.5 (Simplex Safety)**: Under the Simplex architecture with reachability analysis using valid conservative bounds: 
$$\mathbb{P}(\text{Safety Violation}) = 0$$
for all constraints expressible in STL with known physiological bounds.

*Proof sketch*:  The reachability set overapproximates all possible trajectories. If the reachability set does not intersect unsafe states, no trajectory can reach unsafe states. The reflex controller provides additional model-free protection. Full proof in Appendix D.

**Theorem 6.6 (Cold Start Safety)**: Under the hierarchical prior with $\alpha_{strict} = 0.01$: 
$$\mathbb{P}(\text{Day 1 Safety Violation}) \leq 0.01$$
with probability converging to patient-specific $\alpha_{standard} = 0.05$ as $t \to \infty$.

*Proof sketch*: The conservative quantile of the population distribution bounds individual deviation. The relaxation schedule maintains the coverage guarantee as the posterior updates. Full proof in Appendix E.

---

## 7. Experimental Evaluation

### 7.1 Simulation Environment

We evaluate AEGIS 3.0 using the **UVA/Padova Type 1 Diabetes Simulator** (simglucose), an FDA-accepted simulator for testing insulin dosing algorithms [20]. This simulator provides: 
- Physiologically accurate glucose-insulin dynamics
- Virtual patient cohort with inter-patient variability
- Meal and exercise disturbances
- Realistic CGM sensor noise

**Experimental Protocol**:
- **Virtual Patients**: N = 30 (10 children, 10 adolescents, 10 adults)
- **Trial Duration**: 8 weeks per patient (2 weeks run-in, 4 weeks active, 2 weeks washout)
- **Decision Points**: 6 per day (meal times and between-meal periods)
- **Intervention**: Bolus timing suggestions and activity prompts
- **Randomization**: MRT with contextual stratification
- **Random Seeds**: 5 per configuration for statistical power

### 7.2 Baseline Systems

We compare against: 

1. **Standard PID Control**: Classical proportional-integral-derivative controller with population-tuned parameters
2. **Naive RL**: Q-learning with linear function approximation and ε-greedy exploration
3. **JITAI (No Causal)**: Just-in-time adaptive intervention with regression-based effect estimation (no MRT, no causal identification)
4. **Digital Twin Only**: UDE-based prediction with greedy action selection (no causal inference, no safety constraints)
5. **Ablated AEGIS variants**: Testing contribution of each component

### 7.3 Evaluation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Time-in-Range (TIR)** | % of time glucose 70-180 mg/dL | Maximize (↑) |
| **Hypoglycemic Events** | Episodes of glucose < 70 mg/dL | Minimize (↓) |
| **Hyperglycemic Events** | Episodes of glucose > 250 mg/dL | Minimize (↓) |
| **Safety Violations** | Severe hypo (< 54) or hyper (> 300) | Zero (↓) |
| **Adaptation Speed** | Time to reach 90% of asymptotic performance | Minimize (↓) |
| **Cumulative Regret** | Sum of suboptimal action costs | Minimize (↓) |

### 7.4 Main Results

**Table 1:  End-to-End Performance Comparison**

| Method | TIR (%) ↑ | Hypo Events ↓ | Hyper Events ↓ | Safety Violations ↓ |
|--------|-----------|---------------|----------------|---------------------|
| Standard PID | 62.3 ± 4.2 | 3.2 ± 1.1 | 5.4 ± 1.8 | 2.1 ± 0.8 |
| Naive RL | 58.7 ± 6.3 | 4.8 ± 2.3 | 4.1 ± 1.5 | 5.3 ± 2.1 |
| JITAI (No Causal) | 67.1 ± 3.8 | 2.4 ± 0.9 | 4.8 ± 1.4 | 1.4 ± 0.6 |
| Digital Twin Only | 69.4 ± 3.5 | 2.1 ± 0.8 | 3.9 ± 1.2 | 0.8 ± 0.4 |
| **AEGIS 3.0** | **78.2 ± 2.9** | **1.2 ± 0.5** | **2.3 ± 0.8** | **0.0 ± 0.0** |

*Results averaged over 30 virtual patients, 8-week trials, 5 random seeds.  Bold indicates statistically significant improvement (p < 0.01, paired t-test with Bonferroni correction).*

AEGIS 3.0 achieves **25.5% relative improvement** in time-in-range over standard PID control and **16.5% improvement** over the best baseline (Digital Twin Only), with **zero safety violations** compared to all baselines.

### 7.5 Scenario-Specific Results

#### Scenario A: Non-Stationarity ("Flu Shock")

We simulate sudden physiological shift (insulin sensitivity increases 50% at t=24h, simulating acute illness onset).

**Figure 1**:  State estimation comparison during physiological shift

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
| Time-to-Adapt (h) | >24 | 5. 8 ± 1.2 | — |
| Effect Bias | 0.42 ± 0.15 | 0.08 ± 0.04 | 0.00 |

The AC-UKF's innovation-based covariance adaptation detects the residual spike and inflates process noise, achieving near-oracle performance within 6 hours.

#### Scenario B: Circadian Confounding ("Time-of-Day Trap")

Ground truth: τ(morning) = 0.0, τ(evening) = 0.5.  Patient availability is biased toward evening (70% evening, 30% morning), creating spurious treatment-time-of-day correlation.

**Figure 2**: Treatment effect estimation under circadian confounding

```
              Treatment Effect τ(t)
       0.6 ┤                        ── True
           │                  ╭──   ── AEGIS (Harmonic)
       0.5 ┤              ╭───╯     ·· Naive MRT
           │          ╭───╯
       0.4 ┤      ╭───╯
           │  ╭───╯
       0.3 ┤──╯
           │     ·····························
       0.2 ┤           (Naive:  biased estimate)
           │
       0.1 ┤
           │
       0.0 ┼────────────
           │
          -0.1 ┼────┬────┬────┬────┬────┬────
              0    4    8   12   16   20   24  Hour
```

| Metric | Naive MRT | Time-Stratified | Harmonic (AEGIS) |
|--------|-----------|-----------------|------------------|
| Abs Bias (ψ₀) | 0.31 ± 0.08 | 0.18 ± 0.06 | 0.04 ± 0.02 |
| 95% CI Coverage | 0.72 | 0.86 | 0.94 |

AEGIS's Harmonic G-estimation absorbs circadian variation, recovering the true time-varying effect with near-nominal coverage.

#### Scenario C:  Exploration Collapse ("Seldonian Bottleneck")

Ground truth: Optimal action has risk = 5. 5%, safety threshold = 5.0%. The optimal action is repeatedly blocked under standard constrained bandits.

**Figure 3**:  Cumulative regret comparison

```
              Cumulative Regret
      100 ┤                            ── Standard TS (linear)
           │                       ╱
       80 ┤                    ╱
           │                 ╱
       60 ┤              ╱       ── Conservative (sublinear)
           │           ╱        ╱
       40 ┤        ╱        ╱
           │     ╱      ╱         ── AEGIS CTS (sublinear)
       20 ┤  ╱   ╱───────────────
           │╱───
        0 ┼────┬────┬────┬────┬────
           0  200  400  600  800  1000  Rounds
```

| Metric | Standard TS | ε-Greedy | Conservative | CTS (AEGIS) |
|--------|-------------|----------|--------------|-------------|
| Regret (T=1000) | 98.2 ± 12.4 | 67.3 ± 9.1 | 45.2 ± 6.8 | **28.4 ± 4.2** |
| Safety Violations | 0.0 | 2.1 ± 0.8 | 0.0 | 0.0 |
| Posterior Var (blocked) | 0.89 | 0.76 | 0.91 | 0.24 |

CTS maintains posterior updates via counterfactual imputation, preventing the linear regret that plagues Standard TS under safety constraints while maintaining zero violations.

### 7.6 Ablation Study

**Table 2: Ablation Study - Contribution of Each Component**

| Configuration | TIR (%) | Δ from Full | Safety Viol.  |
|---------------|---------|-------------|--------------|
| **Full AEGIS 3.0** | 78.2 | — | 0.0 |
| − Proximal Adjustment | 74.1 | -4.1 | 0.0 |
| − Harmonic G-Estimation | 73.5 | -4.7 | 0.0 |
| − Counterfactual TS | 71.8 | -6.4 | 0.0 |
| − Adaptive Filtering | 75.6 | -2.6 | 0.2 |
| − Cold-Start Safety | 76.4* | -1.8 | 1.3* |
| − Simplex Safety | 72.3† | -5.9 | 4.2† |

*\* Similar asymptotic TIR but 1.3 safety violations concentrated in first week*
*† 4.2 safety violations throughout trial*

Each component contributes to overall performance. The Simplex Safety layer provides the largest safety margin; Counterfactual TS provides the largest efficacy contribution.

### 7.7 Computational Feasibility Analysis

**Table 3: Computational Requirements**

| Component | Time Complexity | Mean Runtime | Max Runtime |
|-----------|-----------------|--------------|-------------|
| Semantic Extraction | O(L·V) | 0.32s | 0.48s |
| AC-UKF Update | O(n³) | 0.008s | 0.012s |
| RBPF Update | O(N·n²) | 0.21s | 0.38s |
| G-Estimation | O(T·K²) | 12.4s (batch) | 18.2s |
| CTS Selection | O(|A|·d) | 0.04s | 0.07s |
| STL Monitoring | O(T·|φ|) | 0.02s | 0.03s |

*Runtime measured on standard computing hardware (Intel i7, 16GB RAM). n=6 states, N=500 particles, L=512 tokens, V=32000 vocabulary, T=1000 timepoints, K=3 harmonics, |A|=5 actions, d=10 parameters.*

All time-critical components (safety monitoring, action selection) complete within clinically acceptable timescales.  Batch operations (G-estimation) run asynchronously without affecting decision latency.

---

## 8. Discussion

### 8.1 Key Findings

1. **Integration Matters**: No single component achieves AEGIS 3.0's performance. The ablation study demonstrates synergistic contributions from each architectural layer.

2. **Causal Inference is Critical**: The 4-6% TIR improvements from Proximal Adjustment and Harmonic G-estimation demonstrate that proper causal identification—not merely prediction—is essential for treatment optimization.

3. **Safety and Efficacy are Complementary**: The Simplex architecture achieves zero safety violations while enabling more aggressive optimization. Safety constraints do not inherently compromise efficacy when properly designed.

4. **Counterfactual Learning Resolves Exploration-Safety Tension**: CTS's counterfactual updates prevent the posterior collapse that plagues standard constrained optimization, achieving sublinear regret without safety compromise.

### 8.2 Limitations

**Proxy Validity**:  Proximal causal inference requires valid negative control proxies. Not all unmeasured confounders admit text-based proxies; some (e.g., genetic variants) leave no narrative trace.  Automated proxy validity verification remains an open problem.

**Simulator Fidelity**: While UVA/Padova is FDA-accepted for insulin algorithm testing, simulation cannot capture all aspects of real-world variability (patient behavior, sensor failures, communication delays). Clinical validation remains necessary.

**Computational Complexity**:  RBPF with sufficient particles for multimodal tracking remains computationally demanding. Current implementation restricts RBPF to 500 particles, potentially limiting fidelity for highly complex distributions.

**Population Prior Quality**: Cold-start safety depends on population prior validity. For rare diseases or novel treatments without historical data, the hierarchical framework provides limited benefit.

**Single Disease Application**:  Evaluation is limited to Type 1 Diabetes.  Generalization to other chronic conditions requires validation. 

### 8.3 Broader Impact and Ethical Considerations

**Potential Benefits**:  AEGIS 3.0 could improve treatment outcomes for millions of patients with chronic
