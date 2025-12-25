import numpy as np
import skfuzzy as fuzz

def fuzzy_risk_level(prob):
    x = np.arange(0, 1.01, 0.01)

    low = fuzz.trimf(x, [0, 0, 0.5])
    medium = fuzz.trimf(x, [0.3, 0.5, 0.7])
    high = fuzz.trimf(x, [0.6, 1, 1])

    low_val = fuzz.interp_membership(x, low, prob)
    mid_val = fuzz.interp_membership(x, medium, prob)
    high_val = fuzz.interp_membership(x, high, prob)

    if high_val > max(low_val, mid_val):
        return "High Risk"
    elif mid_val > low_val:
        return "Medium Risk"
    else:
        return "Low Risk"
