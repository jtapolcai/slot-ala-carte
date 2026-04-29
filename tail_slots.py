"""
TailSlots: data structure grouping attack strings that share the same epoch tail.

For each distinct tail string, TailSlots stores a list of AttackString objects
indexed by head_index (0 = no head, k = 'H^{k-1}A' head).  This allows
the heuristic utility to look up pre-computed utilities for both head and tail
components in O(1) per slot.

Key methods:
    add_attack_string()  -- insert a new attack string under this tail
    eas_utility()        -- probability-weighted expected continuation utility
    get_head_utility() -- monotone list of head utilities indexed by first-A
"""

from platform import node
# from forking_attack import ForkingAttack
from forking_string import count_head_Hs, toChain, attack_str, str_head, get_head_index
from logger import is_debug, log, set_debug
#from forking_string import count_sacrificed
# from saveresults import save_xml
import json
from known_outcome import KnownOutcome
from attack_string import AttackString
from honest_attack import HonestAttack
# from selfish_mixing_attack import SelfishMixingAttack

set_debug(2)
debug = True

def attack_prob(tail,head_index,alpha):
    As = tail.count("A")
    Hs = tail.count("H")
    if head_index>0:
        Hs+=head_index-1
        As+=1
    return pow(alpha, As) * pow(1 - alpha, Hs)


class TailSlots:
    """
    class for storing all nodes (attacks) for a given tail

    for each tail multiple heades could be, which is stored in an array
      the following table summarizes the index in the array:
        head_index 0 -> ''
        head_index 1 -> 'A'
        head_index 2 -> 'HA'
        head_index x -> 'H^{x-1}A' 
    """
    head_weight=0.0

    def __init__(self, tail):
        self.tail = tail  # the tail string
        self.pad = ""
        self.avg_utility = None
        self.attack_strings = {} # dict of nodes, key is head_index

    def max_head(self):
        return max(self.attack_strings.keys()) if self.attack_strings else -1
        
    def get_head(self, head_index):
        """
        returns True if there is an attack string: tail.head 
        """
        if head_index in self.attack_strings:
            return self.attack_strings[head_index]
        return None

    def has_head(self, head_index):
        return head_index in self.attack_strings and self.attack_strings[head_index] is not None

    def get_matching_heades(self, head_index):
        """
        returns the list of all matchnign heades
        """
        ret = []
        for pi , attack_string_obj in self.attack_strings.items(): 
            if attack_string_obj is not None:
                if pi==0:
                    ret.append(attack_string_obj)
                elif pi >= head_index:
                    ret.append(attack_string_obj)
        return ret
    
    def matching_head(self, head_index):
        """
        returns an attack string if there is an attack string that matches tail.head , otherwise None
        """
        if head_index in self.attack_strings and self.attack_strings[head_index] is not None:
            return self.attack_strings[head_index]
        
        for key,value in self.attack_strings.items():
            if key > head_index and value is not None:
                return value

        if 0 in self.attack_strings and self.attack_strings[0] is not None:
             return self.attack_strings[0]
        return None
        
    def get_node_head(self, head_index):
        """
        returns the attack string: tail.head 
        """
        # idx = self.get_index(head_index)
        # return idx in self.attack_strings and self.attack_strings[idx] is not None
        if head_index in self.attack_strings:
             return self.attack_strings[head_index]
        return None

    def add_attack_string(self, astring, parent, nofork_attack_string, head_index=None):
        """
            call this when a new attack string is found
        """
        if head_index is None:
            head_index = get_head_index(parent.get_head())
        #if debug:
        #    self._debug_check_node(node, head_index)
        attack_string_obj = AttackString(astring, parent, head_index)
        updated, new_attack = attack_string_obj.last_update_result

        attack_string_obj.set_nofork(nofork_attack_string)
        self.attack_strings[head_index] = attack_string_obj

        return attack_string_obj, updated, new_attack


    def utility(self, index):
        if index in self.attack_strings and self.attack_strings[index] is not None:
             return self.attack_strings[index].utility
        return 0.0

    def eas_utility(self):
        if self.avg_utility is not None:
            return self.avg_utility
        
        ret = 0
        sum_prob = 0
        prob = HonestAttack.alpha + TailSlots.head_weight
        
        # Sum up for specific heades 1..max
        for index in range(1, self.max_head() + 1):
            utility = 0
            # Simplify lookup logic: use direct access if available and positive, else fallback to utility(0)
            if index in self.attack_strings and self.attack_strings[index] is not None and self.attack_strings[index].utility > 0:
                utility = self.attack_strings[index].utility
            else:
                utility = self.utility(0)

            ret += prob * utility
            sum_prob += prob
            prob *= (1 - HonestAttack.alpha)
            
        if self.attack_strings: # Check if dict is not empty
            ret += (1 - sum_prob) * self.utility(0)
        
        self.avg_utility = ret
        return ret
    
    def get_head_utility(self):
        """
        computes how to adjust utility once a next epoch string is known
        it is list of utility adjustment
        index: the position of the first A in the next epoch string
        ret[i] = best utility when first A in next epoch is at position i.
        The attacker can always fall back to the head_index=0 strategy (no next-epoch A),
        so ret[i] = max(utility(0), utility(i)).  This guarantees ret is non-decreasing.
        """
        max_head = self.max_head()
        base_util = self.utility(0)
        if max_head == 0:
            return [base_util]
        ret = [base_util] * (max_head + 1)
        for i in range(1, max_head + 1):
            ret[i] = max(base_util, self.utility(i))
        return ret
    
    def probability(self):
        """ the probability of the tail, set pad before with EpochUtilityFunction.make_unique_heades() """
        padded=self.pad+self.tail
        As = padded.count("A")
        Hs = padded.count("H")
        return pow(HonestAttack.alpha, As) * pow(1 - HonestAttack.alpha, Hs)
        
        

    def items(self):
        # Return list of (head_str, utility) for nodes with non-zero utility
        # Using sorted to maintain deterministic order (by index, then head string implicitly)
        return [(str_head(i), self.attack_strings[i].utility) for i in sorted(self.attack_strings.keys()) if self.attack_strings[i].utility != 0]

    def __len__(self):
        return len(self.items())

    def __repr__(self):
        ret = f"TailSlots {self.tail} (eas utiltiy: {self.eas_utility()}, {self.items()}): \n"
        # Reconstruct shared view for repr
        all_cns = set()
        for node in self.attack_strings.values():
            if node:
                all_cns.update(node.realization_plan.keys())
        
        for cn in all_cns:
             ret += f"{cn}: ["
             for i in range(self.max_head()+1):
                 node = self.attack_strings.get(i)
                 if node and cn in node.realization_plan:
                      ret += f"{node.realization_plan[cn]} "
                 else:
                      ret += "None "
             ret += "]\n"

        ret+="\nhead utility: "
        for i in range(self.max_head()+1):
            ret+=f"{self.utility(i):.2f}, "
        ret+="\n" 
        return ret
    
if __name__ == "__main__":
    # from avg_utilityfunction import EpochUtilityFunction
    import save_policy
    
    load="attack_list.json"
    # honest = save_policy.load_attack_tree(load)
    
    # attack_strings=EpochUtilityFunction()
    # compute_heuristic_utility_all(attack_strings)
    # attack_strings.avgUtility()
    
    # save_policy.save_utility_function(attack_strings, "saved_utility.json")

