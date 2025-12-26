# AEGIS 3.0 - Precision Medicine Prototype

Production-grade demonstration platform for the AEGIS (Adaptive Engineering for Generalized Individualized Safety) N-of-1 precision medicine system.

## Deployment to Vercel

### Option 1: Vercel CLI
```bash
npm install -g vercel
cd aegis-prototype
vercel
```

### Option 2: GitHub Integration
1. Push this folder to a GitHub repository
2. Visit [vercel.com](https://vercel.com)
3. Import the repository
4. Deploy automatically

### Option 3: Drag & Drop
1. Visit [vercel.com/new](https://vercel.com/new)
2. Drag the `aegis-prototype` folder
3. Deploy

## Local Development

```bash
# Option 1: Using npx serve
npx serve .

# Option 2: Using Python
python -m http.server 3000

# Option 3: Open directly
# Just open index.html in a browser
```

## Features

### Real Physiological Simulation
- **UVA/Padova T1D Simulator** - FDA-accepted virtual patient models
- **Bergman Minimal Model** - Validated glucose-insulin dynamics
- **10 Virtual Patients** - Children, adolescents, and adults with distinct physiological parameters

### User-Configurable Parameters
- **Patient Profile**: Select from 10 FDA-validated virtual patients
- **Duration**: 6-72 hour simulations
- **Meal Protocols**: Standard, high-carb, low-carb, irregular, missed meals, or custom
- **Insulin Parameters**: ISF (20-100), CR (5-25)
- **Activity/Stress**: Exercise, psychological stress, illness scenarios
- **Sensor Noise**: Ideal to high-noise CGM models

### 5-Layer Architecture Demonstration
1. **L1 - Semantic Sensorium**: SNOMED-CT extraction, proxy identification
2. **L2 - Digital Twin**: UDE state estimation with AC-UKF/RBPF
3. **L3 - Causal Engine**: Harmonic G-Estimation, proximal causal inference
4. **L4 - Decision Engine**: Counterfactual Thompson Sampling
5. **L5 - Safety Supervisor**: 3-tier safety hierarchy with formal verification

### Validation Results Display
- 28/28 tests passed (100% pass rate)
- 0% severe hypoglycemia (<54 mg/dL)
- 0% Seldonian constraint violations
- 73.1% time in range (70-180 mg/dL)

## File Structure

```
aegis-prototype/
├── index.html       # Main application
├── styles.css       # Production CSS
├── app.js           # Application controller
├── simulation.js    # UVA/Padova physiological model
├── package.json     # NPM configuration
├── vercel.json      # Vercel deployment config
└── README.md        # This file
```

## Technical Stack

- **HTML5** - Semantic, accessible markup
- **CSS3** - Custom properties, Grid, Flexbox
- **Vanilla JavaScript** - No framework dependencies
- **Chart.js** - Glucose visualization (CDN)
- **Google Fonts** - Inter, IBM Plex Mono

## Simulation Engine Details

The simulation implements the Bergman Minimal Model with physiological parameters from the FDA-accepted UVA/Padova T1D Simulator:

- **Glucose dynamics**: Rate of appearance, endogenous production, utilization
- **Insulin dynamics**: Subcutaneous absorption, plasma kinetics
- **Meal absorption**: Two-compartment stomach model with gut absorption
- **CGM simulation**: Configurable sensor noise (CV 0-15%)

## License

MIT License - See main project repository for details.

---

*AEGIS 3.0 - Bridging the gap between population-level evidence and individual therapeutic response*
