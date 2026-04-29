"""
Myopic per-epoch utility computation for RANDAO attack strings.

Provides AttackString, which computes the expected immediate reward and
forking-outcome PMF for a given attack string and adversary fraction alpha.
The forking threshold T_rho is derived by inverting the no-fork CDF, and
the probability mass is split between forking and non-forking branches.

Main entry points:
    AttackString.compute_utility()  -- full myopic utility (reward - sacrifice)
    AttackString.estimate_utility() -- fast estimate used by the heuristic
    AttackString.get_threshold()    -- compute the integer forking threshold T_rho
"""
from __future__ import annotations
from typing import ClassVar
import math 
from logger import is_debug, log, set_debug
from forking_string import toChain, get_head_index, str_head
from utility_distr import expectedValue, F_function, F_inverse_function, compute_Psi_multi, cdf_distr, shiftDistr, computeMaxDistribution
#from forking_string import generate_forking_realizations, count_sacrificed, count_dropped
from honest_attack import HonestAttack
from selfish_mixing_attack import SelfishMixingAttack
from forking_attack import ForkingAttack
#from known_outcome import KnownOutcome

class AttackString:
    # Set by EpochUtilityFunction.__init__ before any AttackString is created.
    # Accessing this before initialization raises AttributeError intentionally.
    honest_attack: ClassVar[AttackString]
    max_binoms_cache: ClassVar[dict] = {}
    max_cdf_binoms_cache: ClassVar[dict] = {}

    @classmethod
    def get_distribution_of_max_binoms(cls, i, j):
        if i<0 or i>32:
            return 0
        return cls.distribution_of_max_binoms(j)[i]
        
    @classmethod
    def distribution_of_max_binoms(cls, j):
        if j==1:
            return HonestAttack.basic_distr
        if j in cls.max_binoms_cache:
            return cls.max_binoms_cache[j]
        else:
            if j<=1:
                log(1, f"Warning! Invalid j={j} for distribution_of_max_binoms")
                return HonestAttack.basic_distr
            
            # Binary decomposition optimization
            # If j is a power of 2
            if (j & (j - 1)) == 0:
                 half = j // 2
                 prop_distr = cls.distribution_of_max_binoms(half)
                 result_distr = computeMaxDistribution(prop_distr, prop_distr)
            else:
                 # Split into largest power of 2 and remainder
                 largest_pow2 = 1 << (j.bit_length() - 1)
                 remainder = j - largest_pow2
                 dist_pow2 = cls.distribution_of_max_binoms(largest_pow2)
                 dist_rem = cls.distribution_of_max_binoms(remainder)
                 result_distr = computeMaxDistribution(dist_pow2, dist_rem)

            cls.max_binoms_cache[j] = result_distr
            return result_distr

    @classmethod
    def get_cdf_of_max_binoms(cls, i, j):
        if i < 0:
            return 0.0
        if i >= 32:
            return 1.0
        
        if j not in cls.max_cdf_binoms_cache:
             cls.max_cdf_binoms_cache[j] = cdf_distr(cls.distribution_of_max_binoms(j))

        return cls.max_cdf_binoms_cache[j][i]            

    def __init__(self, astring=None, parent=None, head_index=None):
        self.attacks = []
        self.realization_plan = {} # this is a map from chain notation to KnownOutcome
        self.utility_val = 0.0
        self.exp_distr = None
        self.psi = None
        self.no_fork = None
        self.pos= None
        self.head_index = head_index
        self.attack_string: str = ""
        self.selfish_mixing_parent = None
        self.last_update_result = (False, None)
        
        if astring is not None:
            self.last_update, self.last_attack_string_obj = self.add_attack(astring, parent, head_index)

      
    @property
    def utility(self):
        return self.utility_val

    def get_next_decision(self):
        return self.no_fork
    
    def set_nofork(self, nofork_node):
         self.no_fork = nofork_node

    def get_attack_string(self):
        return self.attack_string
        
    def get_head(self):
        if self.head_index is not None:
             return str_head(self.head_index)
        return ""
   
    def get_realizations(self):
        return list(self.realization_plan.values())
    
    def is_honest(self):
        return self.attack_string=='.'

    def get_nofork_utility(self):
        if self.no_fork is not None:
            return self.no_fork.utility
        else:
            log(1, f"Warning! No no-fork node found for attack string {self.attack_string}")
            return 0.0

    def get_exp_immediate_reward(self, immediate_reward, value_r):
        ret = 0.0
        value_r = 0 if value_r is None else value_r
        if self.no_fork is not None and self.no_fork.exp_distr is not None:
            for i in range(33):
                ret += self.no_fork.exp_distr[i] * max(i - value_r, immediate_reward)

        return ret

    def add_attack(self, astring: str, parent: "AttackString | None", head_index: "int | None"):
        """
            Use this function to add a new edge to an attack string
        """
        if parent is not None:
            attack_string = astring + (parent.attack_string or "")
        else:
            attack_string = astring 
        if self.attack_string=="":
            self.attack_string = attack_string
            tail, head = attack_string.rsplit(".", 1)
            self.head_index = get_head_index(head)
            self.pos = len(tail) 
        elif self.attack_string != attack_string:
            log(1, f"Warning! Inconsistent attack string addition: existing {self.attack_string}, new {attack_string}")
        for attack in self.attacks:
            if attack.astring == astring and attack.parent == parent:
                return False, attack  # Attack already exists
        #attack = self.create_attack_obj(astring, parent)
        if  astring=='.' or astring=='':
            attack=HonestAttack(HonestAttack.alpha, self)
            self.exp_distr = HonestAttack.basic_distr
        elif astring=="A":
            attack=SelfishMixingAttack(parent)
            if parent is not None and hasattr(parent, 'attack_string') and parent.attack_string == '.':
                # First selfish-mixing step extends honest baseline.
                self.selfish_mixing_parent = parent
            elif parent and hasattr(parent, 'selfish_mixing_parent'):
                # Chained selfish-mixing: use the direct parent so that
                # AA. -> max(A_distr, A_distr-1), not max(honest, honest-1).
                self.selfish_mixing_parent = parent
        elif astring=="A.":
            attack=SelfishMixingAttack(parent)
            self.selfish_mixing_parent = AttackString.honest_attack
        else:
            attack=ForkingAttack(astring,parent)
        attack.attack_string_obj = self
        if len(self.attacks) == 0:
            self.pos = attack.pos
            for realization in attack.get_realizations():
                if HonestAttack.limit_sacrifice is not None and realization.count_sacrificed() > HonestAttack.limit_sacrifice:
                    continue
                cn = toChain(str(realization))
                self.realization_plan[cn] = realization
            updated=True
            self.attacks.append(attack)
        else:    
            # Merge existing realizations if we already have a node for this head
            self.extend_attack(attack)
        #if updated :
        #    self.compute_utility()            
        return updated, attack

    def extend_attack(self, attack):
        if attack in self.attacks:
            return False
        log(4, f"Extending attack string {self.attack_string} with new attack {attack.astring}")
        attack_head=get_head_index(attack.get_head())
        if self.head_index!=attack_head and attack_head!=0 and self.head_index!=None and attack_head>self.head_index:
            log(1, f"Warning! Inconsistent head index in extending attack string: existing {self.attack_string}, new {attack._get_attack_string()}")
        # Merge existing realizations if we already have a node for this head
        new_realisations=attack.get_realizations()
        updated=False
        for outcome in new_realisations:
            cn = toChain(str(outcome))
            if cn not in self.realization_plan:
                if HonestAttack.limit_sacrifice is not None and outcome.count_sacrificed() > HonestAttack.limit_sacrifice:
                    continue
                self.realization_plan[cn] = outcome
                updated=True
            else:
                old_outcome = self.realization_plan[cn]
                if str(outcome).count('S') > str(old_outcome).count('S'):
                    # Prefer higher-sacrifice branch, but keep stronger continuation metadata.
                    if outcome.next_real is None and old_outcome.next_real is not None:
                        outcome.next_real = old_outcome.next_real
                    elif outcome.next_real is not None and old_outcome.next_real is not None:
                        same_len_weaker_threshold = (
                            outcome.next_real.pos() == old_outcome.next_real.pos()
                            and getattr(outcome.next_real, 'treashold', 0.0) < getattr(old_outcome.next_real, 'treashold', 0.0)
                        )
                        if outcome.next_real.pos() < old_outcome.next_real.pos() or same_len_weaker_threshold:
                            outcome.next_real = old_outcome.next_real
                    if getattr(outcome, 'treashold', 0.0) < getattr(old_outcome, 'treashold', 0.0):
                        outcome.treashold = old_outcome.treashold
                    self.realization_plan[cn] = outcome
                    updated=True
                else:
                    # Keep existing branch, but enrich it if the new one has a better continuation.
                    if old_outcome.next_real is None and outcome.next_real is not None:
                        old_outcome.next_real = outcome.next_real
                        updated=True
                    elif old_outcome.next_real is not None and outcome.next_real is not None:
                        same_len_better_threshold = (
                            old_outcome.next_real.pos() == outcome.next_real.pos()
                            and getattr(old_outcome.next_real, 'treashold', 0.0) < getattr(outcome.next_real, 'treashold', 0.0)
                        )
                        if old_outcome.next_real.pos() < outcome.next_real.pos() or same_len_better_threshold:
                            old_outcome.next_real = outcome.next_real
                            updated=True
                    if getattr(old_outcome, 'treashold', 0.0) < getattr(outcome, 'treashold', 0.0):
                        old_outcome.treashold = outcome.treashold
                        updated=True
        self.attacks.append(attack)

    def __repr__(self):
        return self.attack_string if self.attack_string else "UndefinedAttackString"

    def is_forking_attack(self):
         #return 'H' not in self.attack_string
         return not self.is_honest_attack() and not self.is_selfish_mixing()
    
    def is_honest_attack(self):
         return self.attack_string == "."

    def is_selfish_mixing(self):
        return self.selfish_mixing_parent is not None
 
    def get_all_decision_points(self):
        if self.no_fork is None:
            if 'H' in self.attack_string.split('.')[0]:
                return[0]
            else:
                return []
        ret = self.no_fork.get_all_decision_points()
        ret.insert(0,self.no_fork.pos)   
        return ret

    def compute_utility(self):
        if self.attack_string=='.':
            log(4, f"Compute utility for honest attack string")
            self.exp_distr = HonestAttack.basic_distr
            self.psi = compute_Psi_multi(HonestAttack.basic_distr, 32)
            self.estimate_utility()   
            return
        if not self.is_forking_attack():
            # log(4,f"Compute utility for selfish mixing attack ({self.attack_string})")
            if self.selfish_mixing_parent is None:
                log(1, f"Warning! Parent not found for selfish mixing {self.attack_string}")
                return
            miss_distr=shiftDistr(self.selfish_mixing_parent.exp_distr, -1)
            prop_distr=self.selfish_mixing_parent.exp_distr
            self.exp_distr = computeMaxDistribution(prop_distr, miss_distr)
            self.psi = compute_Psi_multi(self.exp_distr, 32)
            self.estimate_utility()  
            return
        log(4, f"Compute utility for forking attack string ({self.attack_string})")
        debug=False
        self.exp_distr = None
        self.utility_val = 0.0001
        self.thresholds = []
        attack_string = str(self)
        tail_len= len(attack_string.rsplit(".", 1)[0])
        # 1. No-fork properties (nstr)
        if self.no_fork is None:
             log(1, f"Warning! No next distribution (next_decision/nofork) found for forking attack {self.attack_string}")
             return
        
        decision_points =self.get_all_decision_points()
        next_exp_distr = self.no_fork.get_exp_distr()
        next_psi= self.no_fork.get_psi()

        P_nstr = next_exp_distr
        cdf_nstr = cdf_distr(P_nstr)

        if P_nstr is None:
            log(1, f"Warning! No next distribution (exp_distr) found for forking attack {self.attack_string}")
            return
        E_nstr = expectedValue(P_nstr)
        log(4, f"  E_nstr (no fork): {E_nstr:.4f}")
        
        curr_realizations_s = []
        no_fork_segment_len = self.pos - self.no_fork.pos

        aggregated_X = {}  # to speed up computation
        realizations_seen = set()
        for cn, known_realisation_obj in self.realization_plan.items():
            if known_realisation_obj.next_real is not None and len(decision_points)>=2 and\
            known_realisation_obj.next_real.pos() < decision_points[1]:
                log(1, f"Warning! Inconsistent realization length in forking attack string {self.attack_string} for realization {known_realisation_obj.realization_str} ->{known_realisation_obj.next_real.realization_str}")
                #continue

            #c_rho = known_realisation_obj.count_sacrificed(no_fork_segment_len)
            c_rho = known_realisation_obj.count_sacrificed()
            r_rho = known_realisation_obj.count_dropped(no_fork_segment_len)

            realization_string = known_realisation_obj.get_realisation()
            # Debug: per-branch realizations
            if (c_rho, r_rho) not in aggregated_X:
                log(4, f"NOFORK: Realization {realization_string} (cn={cn}) c_rho={c_rho}, r_rho={r_rho} (nofork slots {no_fork_segment_len})")
            else:
                log(4, f"FORK: Realization {realization_string} (cn={cn}) c_rho={c_rho}, r_rho={r_rho} (nofork slots {no_fork_segment_len})")
            curr_realizations_s.append({
                'c_rho': c_rho,
                'r_rho': r_rho,
                'obj': known_realisation_obj,
                'T': 0, 't_rho': 0.0
            })
            if (c_rho, r_rho) in aggregated_X:
                aggregated_X[(c_rho, r_rho)] += 1
            else:
                aggregated_X[(c_rho, r_rho)] = 1
            realizations_seen.add(cn)

        # Now, after thresholds_dict is built, update T in curr_realizations_s and accumulate w_nofork

        # Calculate Thresholds: t_rho = F^{-1}(E + r_rho) + c_rho
        thresholds_dict = {}

        for (c_rho, r_rho), jj in aggregated_X.items():
            t_val = E_nstr + r_rho
            continuous_t = F_inverse_function(P_nstr, t_val) + c_rho
            T_rho = math.floor(continuous_t) + 1
            thresholds_dict[(c_rho, r_rho)] = (continuous_t, T_rho)
            

        # Update T, t_rho, threshold values in curr_realizations_s
        for item in curr_realizations_s:
            c_rho = item['c_rho']
            r_rho = item['r_rho']
            continuous_t, T_rho = thresholds_dict[(c_rho, r_rho)]
            item['T'] = T_rho
            item['t_rho'] = continuous_t
            item['obj'].treashold = continuous_t
            self.thresholds.append((item['obj'], T_rho))

        # Aggregate w_nofork by (c_rho, r_rho) groups
        log_w_nofork = 0.0
        for (c_rho, r_rho), count in aggregated_X.items():
            _, T_rho = thresholds_dict[(c_rho, r_rho)]
            limit_idx = T_rho - 1
            prob = min(HonestAttack.get_cdf_binom(limit_idx), 1.0)
            if prob > 0:
                log_w_nofork += count * math.log(prob)
            else:
                log_w_nofork += count * float('-inf')
        w_nofork = math.exp(log_w_nofork) if log_w_nofork > float('-inf') else 0.0

        # 3. Compute P_astr
        P_astr = [0.0] * 33
        
        # 3a. No-fork outcome weight: P_nofork = Product B(<= T_rho - 1)
        log(4, f"  w_nofork: {w_nofork:.6f}")
        
        total_prob = w_nofork
        for v in range(33):
            if P_nstr[v] > 0:
                P_astr[v] += w_nofork * P_nstr[v]
        
        if len(aggregated_X) >=20:
            log(2, f"Warning! Large number of unique (c_rho,r_rho) pairs {len(aggregated_X)}  in forking attack string {self.attack_string}")

        # 3b. Forking outcomes
        #from faster_compute_util import compute_forking_outcomes
        # TODO: gives different result for: python epoch_utility_function.py -attack AAHAHH -alpha 35 -log 6
        #total_prob = compute_forking_outcomes(self, aggregated_X, thresholds_dict, P_nstr, cdf_nstr, next_psi, P_astr, total_prob)
        total_prob = self._compute_forking_outcomes(aggregated_X, thresholds_dict, P_nstr, cdf_nstr, next_psi, P_astr, total_prob)
        
        if total_prob < 0.999999 or total_prob > 1.000001:
            log(3,f"Warning! total probability {total_prob} for forking string {self.attack_string}")

        self.exp_distr = P_astr
        self.psi = compute_Psi_multi(self.exp_distr, 32)
        self.estimate_utility()
        log(4, f"Estimated utility: {self.utility_val:.6f}")
        if self.utility_val<0:
            log(2,f"Warning! negative utility {self.utility_val} for {self.attack_string}")


    def _compute_forking_outcomes(self, aggregated_X, thresholds_dict,  P_nstr, cdf_nstr, next_psi, P_astr, total_prob):
        """
        Helper for compute_utility: computes forking outcomes and updates P_astr and total_prob.
        Returns updated total_prob.
        """

        for idx_outer, ((c_rho, r_rho), jj) in enumerate(aggregated_X.items()):
            continuous_t, T_rho = thresholds_dict[(c_rho, r_rho)]
            k_rho = r_rho - c_rho
            start_i = max(0, T_rho)
            if start_i > 32:
                continue
            rho_util = 0.0
            w_fork_sum = 0.0
            for i in range(start_i, 33):
                last_limit_val = None
                last_limit_idx = None
                w_fork = AttackString.get_distribution_of_max_binoms(i, jj)
                if w_fork == 0:
                    continue
                # Product over other branches (aggregated_X)
                multiplier = 1.0
                for idx_inner, ((s_other, r_other), count_other_total) in enumerate(aggregated_X.items()):
                    if idx_outer == idx_inner:
                        continue
                    k_other = r_other - s_other
                    T_other = thresholds_dict[(s_other, r_other)][1]
                    count_eff = count_other_total
                    c = k_rho - k_other
                    x_arg = i + k_other
                    if next_psi is not None and c in next_psi:
                        if x_arg < 0:
                            psi_val = next_psi[c][0]
                        elif x_arg > 32:
                            psi_val = next_psi[c][32] + (x_arg - 32)
                        else:
                            psi_val = next_psi[c][x_arg]
                    else:
                        limit_arg = float(x_arg)
                        F_val = F_function(P_nstr, limit_arg)
                        target = F_val + c
                        psi_val = F_inverse_function(P_nstr, target)
                    last_limit_val = psi_val - k_rho
                    if idx_outer > idx_inner:
                        shift=0.001
                    else:
                        shift=-0.001
                    last_limit_idx = math.floor(last_limit_val +  shift)
                    last_limit_idx = max(last_limit_idx, T_other - 1)
                    prob = min(AttackString.get_cdf_of_max_binoms(last_limit_idx, count_eff), 1.0)
                    multiplier *= prob
                    if multiplier == 0:
                        break
                w_fork *= multiplier
                w_fork_sum += w_fork
                if w_fork == 0:
                    continue
                v0 = i - c_rho
                if 0 <= v0 and i <= 32 - k_rho:
                    idx_cdf = i + k_rho
                    if idx_cdf < 0:
                        prob = 0.0
                    elif idx_cdf >= 33:
                        prob = 1.0
                    else:
                        prob = cdf_nstr[idx_cdf]
                    contribution = w_fork * prob
                    P_astr[v0] += contribution
                    rho_util += v0 * contribution
                loop_start = max(0, v0 + 1)
                loop_end = 32 - r_rho
                loop_end = min(32, loop_end)
                if loop_start <= loop_end:
                    for v in range(loop_start, loop_end + 1):
                        idx = v + r_rho
                        if 0 <= idx < 33:
                            p_val = P_nstr[idx]
                            contribution = w_fork * p_val
                            P_astr[v] += contribution
                            rho_util  += v * contribution
                # DEBUG: log boundary indices
                if last_limit_val is not None and last_limit_idx is not None:
                    log(5, f"FORKING DEBUG: c_rho={c_rho}, r_rho={r_rho}, k_rho={k_rho}, T_rho={T_rho}, limit_val={last_limit_val}, limit_idx={last_limit_idx}")
            description=f"The {jj} branches with s={c_rho},r={r_rho}, k={k_rho} (threshold={T_rho})"
            log(4, f"  {description} (prob:{w_fork_sum:.6f}): summing i from {start_i} to 32 util_rho: {rho_util:.6f} ->{rho_util+T_rho:.6f}")
            total_prob += w_fork_sum
        return total_prob

    #def padded_string(self,head_index, padded=False):
    #    pad=""
    #    if padded and hasattr(self, "pad"):
    #        pad=self.pad.replace('A','a').replace('H','h')
    #    return f"{pad}{self.tail}.{str_head(self.get_index(head_index))}" 


    def estimate_utility(self):
        self.utility_val = expectedValue(self.exp_distr, HonestAttack.alpha)

    # Added methods for access
    def get_exp_distr(self):
        return self.exp_distr

    def get_psi(self):
        return self.psi

    # def utility(self, head_index):
    #     idx = self.get_index(head_index)
    #     if idx in self.attack_strings and self.attack_strings[idx] is not None:
    #          return self.attack_strings[idx].utility
    #     return 0.0

    # def set_utility(self, head_index, value):
    #      idx = self.get_index(head_index)
    #      if idx in self.attack_strings and self.attack_strings[idx] is not None:
    #          self.attack_strings[idx].utility = value
    
    def get_nofork_segment(self,head_index):
        if self.attack_string==".":
            return ""
        if self.no_fork is None:
             pos=0
        else:
             pos=self.no_fork.pos
        tail=self.attack_string.rsplit(".", 1)[0]
        if pos==0:
            return tail
        return tail[:-pos]

    # def get_next(self, head_index):
    #     idx = self.get_index(head_index)
    #     if idx in self.attack_strings and self.attack_strings[idx] is not None:
    #          return getattr(self.attack_strings[idx], 'next_decision', None)
    #     return None
    
    # def set_next_decision(self, head_index,tailObj):
    #     idx = self.get_index(head_index)
    #     if idx not in self.attack_strings:
    #         self.attack_strings[idx] = LoadedNode()
        
    #     if self.attack_strings[idx] is not None:
    #         self.attack_strings[idx].next_decision=tailObj.tail
