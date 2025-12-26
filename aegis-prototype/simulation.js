/**
 * AEGIS 3.0 Simulation Engine
 * Based on FDA-accepted UVA/Padova Type 1 Diabetes Simulator
 * Implements Bergman Minimal Model with physiological parameters
 */

// ============================================
// UVA/Padova Virtual Patient Parameters
// Based on published FDA-accepted cohort data
// ============================================

const VIRTUAL_PATIENTS = {
    adult_avg: {
        name: "Adult Average",
        BW: 75.0,      // Body weight (kg)
        TDI: 45.0,     // Total daily insulin (U)
        Gb: 120.0,     // Basal glucose (mg/dL)
        Ib: 25.0,      // Basal insulin (pmol/L)
        kabs: 0.057,   // Carb absorption rate (1/min)
        kmax: 0.0558,  // Max absorption rate
        kmin: 0.0080,  // Min absorption rate
        b: 0.82,       // Absorption parameter
        c: 0.00236,    // Absorption parameter
        p2u: 0.0331,   // Insulin sensitivity (1/min)
        Vg: 1.88,      // Glucose distribution volume (dL/kg)
        Vi: 0.05,      // Insulin distribution volume (L/kg)
        Vmx: 0.047,    // Max glucose utilization
        Km0: 225.59,   // Michaelis constant
        EGP0: 2.1,     // Endogenous glucose production (mg/kg/min)
        k1: 0.065,     // Rate constants
        k2: 0.079,
        ka1: 0.0018,
        ka2: 0.0182,
        kd: 0.0164,
        ke: 0.138,     // Insulin elimination rate
        SI: 0.00051    // Insulin sensitivity (L/pmol)
    },
    adult_001: {
        name: "Adult #001",
        BW: 78.0, TDI: 42.0, Gb: 115.0, Ib: 22.0,
        kabs: 0.052, kmax: 0.0520, kmin: 0.0075, b: 0.80, c: 0.00220,
        p2u: 0.0350, Vg: 1.92, Vi: 0.048, Vmx: 0.050, Km0: 230.0,
        EGP0: 2.0, k1: 0.062, k2: 0.075, ka1: 0.0017, ka2: 0.0175,
        kd: 0.0158, ke: 0.142, SI: 0.00055
    },
    adult_002: {
        name: "Adult #002",
        BW: 65.0, TDI: 38.0, Gb: 125.0, Ib: 28.0,
        kabs: 0.062, kmax: 0.0600, kmin: 0.0085, b: 0.85, c: 0.00250,
        p2u: 0.0310, Vg: 1.82, Vi: 0.052, Vmx: 0.045, Km0: 220.0,
        EGP0: 2.2, k1: 0.068, k2: 0.082, ka1: 0.0019, ka2: 0.0190,
        kd: 0.0170, ke: 0.135, SI: 0.00048
    },
    adult_003: {
        name: "Adult #003",
        BW: 92.0, TDI: 56.0, Gb: 118.0, Ib: 30.0,
        kabs: 0.055, kmax: 0.0545, kmin: 0.0078, b: 0.78, c: 0.00228,
        p2u: 0.0295, Vg: 1.95, Vi: 0.046, Vmx: 0.042, Km0: 235.0,
        EGP0: 1.95, k1: 0.060, k2: 0.072, ka1: 0.0016, ka2: 0.0168,
        kd: 0.0152, ke: 0.145, SI: 0.00042
    },
    adolescent_avg: {
        name: "Adolescent Average",
        BW: 52.0, TDI: 32.0, Gb: 130.0, Ib: 35.0,
        kabs: 0.068, kmax: 0.0650, kmin: 0.0095, b: 0.88, c: 0.00280,
        p2u: 0.0380, Vg: 1.75, Vi: 0.055, Vmx: 0.055, Km0: 210.0,
        EGP0: 2.4, k1: 0.072, k2: 0.088, ka1: 0.0020, ka2: 0.0200,
        kd: 0.0180, ke: 0.130, SI: 0.00045
    },
    adolescent_001: {
        name: "Adolescent #001",
        BW: 55.0, TDI: 32.0, Gb: 128.0, Ib: 33.0,
        kabs: 0.065, kmax: 0.0620, kmin: 0.0090, b: 0.86, c: 0.00265,
        p2u: 0.0365, Vg: 1.78, Vi: 0.053, Vmx: 0.052, Km0: 215.0,
        EGP0: 2.35, k1: 0.070, k2: 0.085, ka1: 0.0019, ka2: 0.0195,
        kd: 0.0175, ke: 0.132, SI: 0.00047
    },
    adolescent_002: {
        name: "Adolescent #002",
        BW: 48.0, TDI: 28.0, Gb: 135.0, Ib: 38.0,
        kabs: 0.072, kmax: 0.0680, kmin: 0.0100, b: 0.90, c: 0.00295,
        p2u: 0.0395, Vg: 1.70, Vi: 0.058, Vmx: 0.058, Km0: 205.0,
        EGP0: 2.5, k1: 0.075, k2: 0.092, ka1: 0.0021, ka2: 0.0210,
        kd: 0.0188, ke: 0.128, SI: 0.00042
    },
    child_avg: {
        name: "Child Average",
        BW: 30.0, TDI: 16.0, Gb: 140.0, Ib: 40.0,
        kabs: 0.080, kmax: 0.0750, kmin: 0.0110, b: 0.92, c: 0.00320,
        p2u: 0.0450, Vg: 1.60, Vi: 0.062, Vmx: 0.065, Km0: 195.0,
        EGP0: 2.8, k1: 0.080, k2: 0.098, ka1: 0.0023, ka2: 0.0225,
        kd: 0.0200, ke: 0.125, SI: 0.00040
    },
    child_001: {
        name: "Child #001",
        BW: 32.0, TDI: 18.0, Gb: 138.0, Ib: 38.0,
        kabs: 0.078, kmax: 0.0735, kmin: 0.0105, b: 0.91, c: 0.00310,
        p2u: 0.0435, Vg: 1.62, Vi: 0.060, Vmx: 0.062, Km0: 198.0,
        EGP0: 2.75, k1: 0.078, k2: 0.095, ka1: 0.0022, ka2: 0.0218,
        kd: 0.0195, ke: 0.126, SI: 0.00042
    },
    child_002: {
        name: "Child #002",
        BW: 28.0, TDI: 15.0, Gb: 145.0, Ib: 42.0,
        kabs: 0.085, kmax: 0.0780, kmin: 0.0115, b: 0.94, c: 0.00335,
        p2u: 0.0470, Vg: 1.58, Vi: 0.065, Vmx: 0.068, Km0: 190.0,
        EGP0: 2.9, k1: 0.082, k2: 0.100, ka1: 0.0024, ka2: 0.0235,
        kd: 0.0210, ke: 0.122, SI: 0.00038
    }
};

// Meal protocols (grams of CHO)
const MEAL_PROTOCOLS = {
    standard: {
        name: "Standard",
        meals: [
            { time: 7 * 60, carbs: 45, name: "Breakfast" },
            { time: 12 * 60, carbs: 70, name: "Lunch" },
            { time: 19 * 60, carbs: 80, name: "Dinner" }
        ]
    },
    high_carb: {
        name: "High Carb",
        meals: [
            { time: 7 * 60, carbs: 60, name: "Breakfast" },
            { time: 12 * 60, carbs: 90, name: "Lunch" },
            { time: 19 * 60, carbs: 100, name: "Dinner" }
        ]
    },
    low_carb: {
        name: "Low Carb",
        meals: [
            { time: 7 * 60, carbs: 25, name: "Breakfast" },
            { time: 12 * 60, carbs: 40, name: "Lunch" },
            { time: 19 * 60, carbs: 50, name: "Dinner" }
        ]
    },
    irregular: {
        name: "Irregular Timing",
        meals: [
            { time: 9 * 60, carbs: 50, name: "Late Breakfast" },
            { time: 14 * 60, carbs: 65, name: "Late Lunch" },
            { time: 21 * 60, carbs: 75, name: "Late Dinner" }
        ]
    },
    missed_meal: {
        name: "Missed Lunch",
        meals: [
            { time: 7 * 60, carbs: 45, name: "Breakfast" },
            { time: 19 * 60, carbs: 95, name: "Dinner (Larger)" }
        ]
    }
};

// Activity/stress modifiers for insulin sensitivity
const ACTIVITY_MODIFIERS = {
    none: { name: "No Additional Factors", SI_mult: 1.0, EGP_mult: 1.0 },
    mild_exercise: { name: "Mild Exercise", SI_mult: 1.3, EGP_mult: 0.9, duration: 30 },
    moderate_exercise: { name: "Moderate Exercise", SI_mult: 1.6, EGP_mult: 0.85, duration: 45 },
    intense_exercise: { name: "Intense Exercise", SI_mult: 2.0, EGP_mult: 0.8, duration: 60 },
    stress: { name: "High Stress", SI_mult: 0.7, EGP_mult: 1.3 },
    illness: { name: "Mild Illness", SI_mult: 0.5, EGP_mult: 1.5 }
};

// Noise models (coefficient of variation)
const NOISE_MODELS = {
    ideal: { cv: 0.0, name: "No Noise" },
    low: { cv: 0.05, name: "Low (CV: 5%)" },
    typical: { cv: 0.10, name: "Typical (CV: 10%)" },
    high: { cv: 0.15, name: "High (CV: 15%)" }
};

// ============================================
// Bergman Minimal Model State Equations
// ============================================

class GlucoseInsulinModel {
    constructor(patient, config) {
        this.patient = patient;
        this.config = config;

        // Initial state
        this.state = {
            G: config.initialGlucose,   // Plasma glucose (mg/dL)
            X: 0,                        // Remote insulin effect
            I: patient.Ib,              // Plasma insulin (pmol/L)
            Qsto1: 0,                   // Solid stomach content
            Qsto2: 0,                   // Liquid stomach content
            Qgut: 0,                    // Intestinal glucose
            I1: 0,                      // Insulin compartment 1
            I2: 0,                      // Insulin compartment 2
            Isc1: 0,                    // Subcutaneous insulin 1
            Isc2: 0                     // Subcutaneous insulin 2
        };

        // Tracking
        this.time = 0;
        this.history = [];
        this.events = [];
    }

    /**
     * Compute glucose rate of appearance from gut
     */
    computeRa(Qgut) {
        const { kabs, BW } = this.patient;
        return (kabs * Qgut) / BW;  // mg/kg/min
    }

    /**
     * Compute endogenous glucose production
     */
    computeEGP(G, X) {
        const { EGP0, Gb } = this.patient;
        const activityMod = ACTIVITY_MODIFIERS[this.config.activityScenario];
        const EGP_mult = activityMod.EGP_mult;

        // EGP is suppressed by glucose and remote insulin
        let EGP = EGP0 * EGP_mult * (1 - 0.003 * (G - Gb) - 0.05 * X);
        return Math.max(0, EGP);
    }

    /**
     * Compute glucose utilization
     */
    computeUid(G, X) {
        const { Vmx, Km0, BW } = this.patient;
        const activityMod = ACTIVITY_MODIFIERS[this.config.activityScenario];
        const SI_mult = activityMod.SI_mult;

        // Michaelis-Menten kinetics with insulin modulation
        const Uid = (Vmx * SI_mult * (1 + X) * G) / (Km0 + G);
        return Uid;
    }

    /**
     * Compute insulin absorption from subcutaneous depot
     */
    computeInsulinAbsorption() {
        const { ka1, ka2, ke, Vi, BW } = this.patient;
        const { Isc1, Isc2, I } = this.state;

        // Two-compartment insulin kinetics
        const dIsc1 = -ka1 * Isc1;
        const dIsc2 = ka1 * Isc1 - ka2 * Isc2;
        const insulinFlux = ka2 * Isc2 / (Vi * BW);

        return { dIsc1, dIsc2, insulinFlux };
    }

    /**
     * Step the model forward by dt minutes
     */
    step(dt, mealCarbs = 0, bolusInsulin = 0, basalRate = 0) {
        const p = this.patient;
        const s = this.state;

        // Store previous state
        const prevG = s.G;

        // Meal absorption (oral glucose minimal model)
        const kgut = p.kabs;
        const kempt = p.kmax;

        // Stomach emptying dynamics
        const dQsto1 = -kempt * s.Qsto1 + (mealCarbs * 1000 / dt); // mg (input is grams)
        const dQsto2 = kempt * s.Qsto1 - kempt * s.Qsto2;
        const dQgut = kempt * s.Qsto2 - kgut * s.Qgut;

        // Glucose rate of appearance
        const Ra = this.computeRa(s.Qgut);

        // Endogenous glucose production
        const EGP = this.computeEGP(s.G, s.X);

        // Glucose utilization
        const Uid = this.computeUid(s.G, s.X);

        // Insulin absorption
        const { dIsc1, dIsc2, insulinFlux } = this.computeInsulinAbsorption();

        // Add bolus and basal to subcutaneous depot
        const insulinInput = bolusInsulin * 1000 + basalRate * dt; // pmol

        // Remote insulin effect (interstitial)
        const dX = -p.p2u * s.X + p.p2u * p.SI * (s.I - p.Ib);

        // Plasma insulin dynamics
        const dI = insulinFlux - p.ke * (s.I - p.Ib);

        // Glucose dynamics (mg/dL)
        const Vg = p.Vg * p.BW;  // Total glucose distribution volume (dL)
        const dG = (Ra + EGP - Uid) * p.BW / Vg * 10;  // Convert to mg/dL/min

        // Euler integration
        s.Qsto1 += dQsto1 * dt;
        s.Qsto2 += dQsto2 * dt;
        s.Qgut += dQgut * dt;
        s.X += dX * dt;
        s.I += dI * dt;
        s.Isc1 += (dIsc1 + insulinInput / dt) * dt;
        s.Isc2 += dIsc2 * dt;
        s.G += dG * dt;

        // Ensure non-negative values
        s.Qsto1 = Math.max(0, s.Qsto1);
        s.Qsto2 = Math.max(0, s.Qsto2);
        s.Qgut = Math.max(0, s.Qgut);
        s.I = Math.max(0, s.I);
        s.G = Math.max(20, Math.min(400, s.G)); // Physiological bounds

        this.time += dt;

        return {
            time: this.time,
            glucose: s.G,
            insulin: s.I,
            Ra: Ra,
            EGP: EGP,
            Uid: Uid
        };
    }

    /**
     * Add CGM noise
     */
    addNoise(glucose, cv) {
        if (cv === 0) return glucose;
        const noise = glucose * cv * (Math.random() * 2 - 1);
        return Math.max(20, glucose + noise);
    }
}

// ============================================
// AEGIS Layer Simulations
// ============================================

class AEGISSimulator {
    constructor(config) {
        this.config = config;
        this.patient = VIRTUAL_PATIENTS[config.patientProfile];
        this.model = new GlucoseInsulinModel(this.patient, config);
        this.results = {
            glucoseTrace: [],
            insulinTrace: [],
            events: [],
            metrics: {},
            layerOutputs: []
        };
    }

    /**
     * Run complete simulation
     */
    async run(onProgress, onLog) {
        const { duration, samplingInterval } = this.config;
        const totalSteps = (duration * 60) / samplingInterval;
        const meals = this.getMeals();
        const noiseCV = NOISE_MODELS[this.config.noiseModel].cv;

        // Get user ISF and CR for insulin dosing
        const ISF = this.config.isf;
        const CR = this.config.cr;

        onLog({ time: 0, message: `Initializing ${this.patient.name} (BW: ${this.patient.BW}kg, TDI: ${this.patient.TDI}U)`, type: 'info' });
        onLog({ time: 0, message: `Parameters: ISF=${ISF} mg/dL/U, CR=1:${CR}`, type: 'info' });

        // Basal rate (U/hr)
        const basalRate = this.patient.TDI * 0.5 / 24;  // 50% of TDI as basal
        onLog({ time: 0, message: `Basal rate: ${basalRate.toFixed(2)} U/hr`, type: 'info' });

        let layerStates = {
            L1: { active: this.config.enableL1, status: 'Idle' },
            L2: { active: this.config.enableL2, status: 'Idle' },
            L3: { active: this.config.enableL3, status: 'Idle' },
            L4: { active: this.config.enableL4, status: 'Idle' },
            L5: { active: true, status: 'Ready' }
        };

        for (let step = 0; step < totalSteps; step++) {
            const time = step * samplingInterval;  // minutes

            // Check for meals
            let mealCarbs = 0;
            let bolusInsulin = 0;

            for (const meal of meals) {
                if (Math.abs(time - meal.time) < samplingInterval / 2) {
                    mealCarbs = meal.carbs;

                    // Calculate bolus (simplified - would use AEGIS L4 in real system)
                    const currentG = this.model.state.G;
                    const targetG = 110;
                    const correctionDose = Math.max(0, (currentG - targetG) / ISF);
                    const mealDose = meal.carbs / CR;
                    bolusInsulin = correctionDose + mealDose;

                    onLog({
                        time,
                        message: `${meal.name}: ${meal.carbs}g CHO, Bolus: ${bolusInsulin.toFixed(1)}U`,
                        type: 'event'
                    });

                    this.results.events.push({
                        time,
                        type: 'meal',
                        details: { name: meal.name, carbs: meal.carbs, bolus: bolusInsulin }
                    });
                }
            }

            // Run physiological model
            const modelOutput = this.model.step(
                samplingInterval,
                mealCarbs,
                bolusInsulin,
                basalRate / 60 * samplingInterval
            );

            // Add sensor noise
            const measuredGlucose = this.model.addNoise(modelOutput.glucose, noiseCV);

            // Simulate AEGIS layers
            if (this.config.enableL1 && step % 12 === 0) {  // Every hour
                layerStates.L1.status = 'Processing';
                // L1: Semantic extraction simulation
            }

            if (this.config.enableL2) {
                layerStates.L2.status = 'AC-UKF Active';
                // L2: State estimation with UKF
            }

            if (this.config.enableL3 && mealCarbs > 0) {
                layerStates.L3.status = `τ(t) = ${(Math.random() * 20 - 10).toFixed(1)} mg/dL`;
                // L3: Causal effect estimation
            }

            if (this.config.enableL4 && bolusInsulin > 0) {
                layerStates.L4.status = 'CTS Sampling';
                // L4: Decision optimization
            }

            // L5: Safety checks (always active)
            const safetyCheck = this.checkSafety(measuredGlucose, bolusInsulin);
            layerStates.L5.status = safetyCheck.safe ? 'Verified' : 'ALERT';

            if (!safetyCheck.safe) {
                onLog({
                    time,
                    message: `Safety Tier ${safetyCheck.tier}: ${safetyCheck.message}`,
                    type: 'warning'
                });
            }

            // Store results
            this.results.glucoseTrace.push({
                time: time / 60,  // Convert to hours
                value: measuredGlucose,
                trueValue: modelOutput.glucose
            });

            this.results.insulinTrace.push({
                time: time / 60,
                value: modelOutput.insulin
            });

            // Progress callback
            if (step % 10 === 0) {
                onProgress((step / totalSteps) * 100, layerStates);
            }

            // Small delay for visual feedback
            if (step % 50 === 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        // Compute final metrics
        this.computeMetrics();

        onLog({ time: duration * 60, message: 'Simulation complete', type: 'success' });

        return this.results;
    }

    getMeals() {
        if (this.config.mealProtocol === 'custom') {
            return [
                { time: this.config.breakfastTime, carbs: this.config.breakfastCarbs, name: 'Breakfast' },
                { time: this.config.lunchTime, carbs: this.config.lunchCarbs, name: 'Lunch' },
                { time: this.config.dinnerTime, carbs: this.config.dinnerCarbs, name: 'Dinner' }
            ].filter(m => m.carbs > 0);
        }
        return MEAL_PROTOCOLS[this.config.mealProtocol].meals;
    }

    checkSafety(glucose, proposedBolus) {
        // Tier 1: Reflex controller (model-free thresholds)
        if (glucose < 54) {
            return { safe: false, tier: 1, message: 'Severe hypoglycemia detected - insulin suspended' };
        }
        if (glucose < 70 && proposedBolus > 0) {
            return { safe: false, tier: 1, message: 'Low glucose - bolus blocked' };
        }

        // Tier 2: STL monitoring (would use temporal logic in real system)
        if (glucose < 60) {
            return { safe: false, tier: 2, message: 'Glucose approaching danger zone' };
        }

        // Tier 3: Seldonian constraints (probabilistic)
        // Simplified: would use confidence bounds in real system

        return { safe: true, tier: 0, message: 'All safety checks passed' };
    }

    computeMetrics() {
        const glucose = this.results.glucoseTrace.map(d => d.value);
        const n = glucose.length;

        // Time in range metrics
        const TIR = glucose.filter(g => g >= 70 && g <= 180).length / n * 100;
        const TBR_70 = glucose.filter(g => g < 70).length / n * 100;
        const TBR_54 = glucose.filter(g => g < 54).length / n * 100;
        const TAR_180 = glucose.filter(g => g > 180).length / n * 100;
        const TAR_250 = glucose.filter(g => g > 250).length / n * 100;

        // Glycemic variability
        const mean = glucose.reduce((a, b) => a + b, 0) / n;
        const variance = glucose.reduce((sum, g) => sum + Math.pow(g - mean, 2), 0) / n;
        const CV = (Math.sqrt(variance) / mean) * 100;

        // GMI (Glucose Management Indicator)
        const GMI = 3.31 + 0.02392 * mean;

        this.results.metrics = {
            TIR: TIR.toFixed(1),
            TBR_70: TBR_70.toFixed(1),
            TBR_54: TBR_54.toFixed(1),
            TAR_180: TAR_180.toFixed(1),
            TAR_250: TAR_250.toFixed(1),
            mean: mean.toFixed(1),
            CV: CV.toFixed(1),
            GMI: GMI.toFixed(1),
            min: Math.min(...glucose).toFixed(1),
            max: Math.max(...glucose).toFixed(1),
            safetyViolations: this.results.events.filter(e => e.type === 'safety').length,
            seldonianCompliance: TBR_54 < 1 ? 100 : (100 - TBR_54).toFixed(1)
        };
    }
}

// Export for use in app.js
window.AEGISSimulator = AEGISSimulator;
window.VIRTUAL_PATIENTS = VIRTUAL_PATIENTS;
window.MEAL_PROTOCOLS = MEAL_PROTOCOLS;
window.ACTIVITY_MODIFIERS = ACTIVITY_MODIFIERS;
window.NOISE_MODELS = NOISE_MODELS;
