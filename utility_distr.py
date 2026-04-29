"""
Probability distribution helpers for RANDAO slot-count analysis.

Provides PMF/CDF computation, distribution shifting, max-of-independent-
binomials convolution, and the F / F_inverse functions used by the threshold
formula.  Also contains the avg_utility lookup table which gives the expected
per-epoch slot gain for an honest adversary at each alpha value.
"""
import math

debug = False

def expectedValue(distr, alpha=0.0):
    ret = -alpha * 32
    if distr is None:
        return ret
    for i in range(len(distr)):
        ret += i * distr[i]
    return ret

def mixtureDistribution(distr1, distr2, prob):
    ret = []
    length = max(len(distr1), len(distr2))
    for i in range(length):
        val1 = distr1[i] if i < len(distr1) else 0.0
        val2 = distr2[i] if i < len(distr2) else 0.0
        ret.append(prob * val1 + (1 - prob) * val2)
    return ret

def shiftDistr(distr, shift):
    if shift == 0:
        return distr
    if distr is None:
        return None

    ret = [0.0] * 33
    shift_int = int(shift)
    shift_frac = shift - shift_int

    for i in range(33):
        pos1 = min(max(i + shift_int, 0), 32)
        pos2 = min(pos1 + 1, 32)
        if pos1 < 33:
            ret[pos1] += distr[i] * (1 - shift_frac)
        if pos2 < 33:
            ret[pos2] += distr[i] * shift_frac
    return ret

def checkDistr(distr):
    if debug:
        sum_distr = sum(distr)
        if sum_distr < 0.99999999 or sum_distr > 1.000000001:
            print(f"Warning! sum is {sum_distr} for distr={distr}")

def computeMaxDistribution(distr1, distr2, alpha=None):
    """ runs in O(n) time where n=33
    computes the distribution of max(X1,X2) where X1 ~ distr1 and X2 ~ distr2
    """
    if distr1 is None and distr2 is None:
        if alpha is not None:
            return None, 0
        else:
            return None
    if distr1 is None:
        if alpha is not None:
            utility = sum(i * val for i, val in enumerate(distr2))
            return distr2, utility - 32 * alpha
        else:
            return distr2
    if distr2 is None:
        if alpha is not None:
            utility = sum(i * val for i, val in enumerate(distr1))
            return distr1, utility - 32 * alpha
        else:
            return distr1           
    
    cdf1 = cdf_distr(distr1)
    cdf2 = cdf_distr(distr2)
    new_distr = []
    utility = 0
    for i in range(33):
        prob = distr1[i] * cdf2[i] + distr2[i] * (cdf1[i-1] if i > 0 else 0)
        new_distr.append(prob)
        utility += i * prob

    if alpha is not None:
        return new_distr, utility - 32 * alpha
    else:
        return new_distr

def cdf_distr(distr):
    cdf = [0.0] * 33

    if len(distr) > 0:
        cdf[0] = distr[0]

        for val in range(1, 33):
            if val < len(distr):
                cdf[val] = cdf[val - 1] + distr[val]
            else:
                cdf[val] = cdf[val - 1]  # Stay constant

        cdf[-1] = 1.0  # Ensure last value is exactly 1.0

    return cdf

def F_function(distr, val):
    if distr is None:
        return 0.0
    
    cdf = cdf_distr(distr)
    # res = expectedValue(distr) # Originally it was adding expected value, let's verify if that's correct from original file.
    # From heuristic_utility.py: F(0) = E[X]. Yes.
    
    res = 0
    for i in range(len(distr)):
        res += i * distr[i]
    
    if val >= 0:
        limit_i = int(math.floor(val))
        for i in range(1, limit_i + 1):
            if i - 1 < 33:
                res += cdf[i - 1]
            else:
                res += 1.0 
        
        frac = val - limit_i
        idx = limit_i
        if idx < 33:
            slope = cdf[idx]
        else:
            slope = 1.0
        res += frac * slope
    return res

def F_inverse_function(distr, val):
    if distr is None:
        return 0.0
    
    cdf = cdf_distr(distr)
    # E = expectedValue(distr)
    current_F = 0
    for i in range(len(distr)):
        current_F += i * distr[i]
    
    if val >= current_F:
        prev_x = 0.0
        for i in range(1, 34):
            slope_idx = i - 1
            if slope_idx < 33:
                slope = cdf[slope_idx]
            else:
                slope = 1.0

            next_F = current_F + slope
            if val <= next_F:
                if slope < 1e-9: return float(i) 
                return prev_x + (val - current_F) / slope
            current_F = next_F
            prev_x = float(i)
        
        return prev_x + (val - current_F)

    else:
        return 0.0

def compute_Psi_multi(distr, c_limit):
    psi_dict = {}
    n = len(distr) - 1
    for c in range(-c_limit, c_limit + 1):
        Psi_c = []
        for i in range(n + 1):
            F_i = F_function(distr, float(i))
            target = F_i + c
            x = F_inverse_function(distr, target)
            Psi_c.append(x)
        psi_dict[c] = Psi_c
    return psi_dict

def avg_utility(alpha):
    table = {
        0.05: 0.0507,
        0.10: 0.1023,
        0.15: 0.1551,
        0.20: 0.2100,
        0.25: 0.2680,
        0.30: 0.3343,
        0.35: 0.4436,
        0.40: 0.5308,
        0.45: 0.6000
    }
    keys = sorted(table.keys())
    if alpha in table:
        return table[alpha]
    if alpha < keys[0]:
        return table[keys[0]]
    if alpha > keys[-1]:
        return table[keys[-1]]
    # Linear interpolation
    for i in range(1, len(keys)):
        if keys[i-1] < alpha < keys[i]:
            x0, y0 = keys[i-1], table[keys[i-1]]
            x1, y1 = keys[i], table[keys[i]]
            return y0 + (y1 - y0) * (alpha - x0) / (x1 - x0)
    # fallback
    return None