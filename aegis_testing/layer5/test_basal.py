"""Quick script to find the correct simglucose basal rate."""
import sys
sys.path.insert(0, ".")
from sim_utils import simulate_patient, clinical_metrics
import numpy as np

meals = [(360, 45), (720, 70), (1080, 80)]
for b in [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]:
    trace = simulate_patient("adolescent#001", 24, b, meals, 42)
    m = clinical_metrics(trace)
    mn = m["mean"]
    mi = m["min"]
    mx = m["max"]
    tir = m["tir"]
    tbr = m["tbr_54"]
    print(f"basal={b:.3f}  mean={mn:.0f}  min={mi:.0f}  max={mx:.0f}  tir={tir:.0f}%  tbr54={tbr:.0f}%")
