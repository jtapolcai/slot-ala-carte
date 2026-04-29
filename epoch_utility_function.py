"""
Heuristic multi-epoch utility function for RANDAO attack policy.

Uses a pygtrie.CharTrie keyed on reversed tail strings for O(k) lookup of
pre-computed attack string utilities.  When a matching tail is not found,
new attack strings are generated on the fly (fly mode) and inserted into
the trie.

The decomposition follows the paper's heuristic:
    utility = u_head(head) + u_tail(tail)  +  immediate_reward

where u_head is the best continuation utility indexed by the first-A
position in the next epoch, and u_tail is the probability-weighted
expected-value sum over the known longest matching tails (D(tail)).

Main class: EpochUtilityFunction
    compute_utility(head_utility, outcome, sacrifice)
    avg_utility(alpha)  -- expected slot gain for honest strategy
"""

import json
import pygtrie
from typing import cast

from forking_string import (get_head_index, count_head_Hs, findLastAbeforeH,
                            is_attack_string_param_type,
                            max_forking_string_length,create_two_epoch_string)
from logger import is_debug, log, set_debug
from tail_slots import TailSlots, get_head_index,attack_prob,str_head 
import save_policy
from attack_string import AttackString
from honest_attack import HonestAttack
from selfish_mixing_attack import SelfishMixingAttack
from forking_attack import ForkingAttack
from decompose_attack_string import decompose_longest_attack_dynamic
from find_longest_attack_string import find_longest_attack

set_debug(2)
debug = False

class EpochUtilityFunction:
    """
    The utility function of an Extended Attack string 
    """
    utility_multiplier= 1

    def __init__(self, alpha, filter=""):
        """
        Initialise the utility function for a given adversary fraction.

        Parameters:
            alpha  -- adversary stake fraction (0.0–1.0), e.g. 0.35
            filter -- optional string to restrict the attack string set
                      (e.g. 'no_weak_forking', 'selfish_mixing')
        """
        self.filter = filter
        self.fly=True
        self.tail_trie = pygtrie.CharTrie() 
        HonestAttack.alpha = alpha
        self.avg_utility_ = 0.0
        HonestAttack.node_list = []
        honest_attack_tail = TailSlots("")
        self.honest_attack = honest_attack_tail.add_attack_string('.',None,None,0)[0]
        AttackString.honest_attack = self.honest_attack
        self.tail_trie[""] = honest_attack_tail
    

    def compute_utility(self, head_utility, outcome, sacrifice):
        """
        Heuristic utility decomposition used by the attack policy.

        Returns a pair:
          - continuation_utility: u_head(c_rho) + u_tail(c_rho)
          - immediate_reward: x_rho (count of A in epoch e+2 realization)

        The external weight w (utility_multiplier) is applied once at the
        policy-evaluation layer (Realization.outcome).
        """
        immediate_reward = outcome.count("A")
        continuation_utility = 0.0

        # When w=0 we use pure myopic mode; continuation term is disabled.
        if EpochUtilityFunction.utility_multiplier > 0:
            firstA = count_head_Hs(outcome) + 1
            # u_head: continuation value that depends on the first adversarial slot position.
            if len(head_utility) > firstA:
                pref_utility = head_utility[0]
                for i in range(firstA, len(head_utility)):
                    if head_utility[i] != None and pref_utility < head_utility[i]:
                        pref_utility = head_utility[i]
                #pref_utility = head_utility[firstA] - (1 - firstA * HonestAttack.alpha)
            elif len(head_utility) > 0 and head_utility[0] != None:
                pref_utility = head_utility[0]
            else:
                pref_utility = 0

            # u_tail: continuation value from longest tail match of c_rho.
            if self.fly:
                self.create_matching_attack_strings(outcome)
            tailObj = cast(TailSlots, self.tail_trie.longest_prefix(outcome[::-1]).value)
            eas_utility = tailObj.eas_utility()
            continuation_utility = pref_utility + eas_utility
            log(4, f"Continuation utility is {continuation_utility:.2f} = head:{pref_utility:.2f} + tail:{eas_utility:.2f}; immediate reward:{immediate_reward:.0f} sacrifice:{-sacrifice} normalize:{-HonestAttack.alpha*32}")

        return continuation_utility, immediate_reward


    def get_node_attack_string(self, attack_string):
        """
        Return (AttackString, TailSlots) for an exact 'tail.head' key.
        Returns (None, None) if the key is not in the trie.
        """
        tail, head_part = attack_string.rsplit(".", 1)
        head_index = get_head_index(head_part)
        rev_tail = tail[::-1]
        if rev_tail not in self.tail_trie:
            return None, None
        tailObj = cast(TailSlots, self.tail_trie[rev_tail])
        if tailObj.tail != tail:
            return None, None
        return tailObj.get_head(head_index), tailObj

    def is_properattack_string(self, tail, head_index, filter=None):
        """
        Return True if (tail, head_index) is a valid attack string
        under the current or explicitly supplied filter.
        """
        if filter is None:
            filter = self.filter
        return is_attack_string_param_type(tail, head_index, filter)

    def get_attack_string(self, attack_string, exact_matching=True):
        """
        returns the attack string object and tail object
        """
        tail, head = attack_string.rsplit(".", 1)
        head_index = get_head_index(head)
        reversed_tail=tail[::-1]
        if reversed_tail in self.tail_trie:
            tailObj = cast(TailSlots, self.tail_trie[reversed_tail])
            if exact_matching:
                return tailObj.get_head(head_index), tailObj
            else:
                return tailObj.matching_head(head_index), tailObj
        return None, None
    
    # def get_matching_attack_string(self, attack_string):
    #     tail, head = attack_string.split(".")
    #     head_index = get_head_index(head)
    #     reversed_tail=tail[::-1]
    #     if reversed_tail in self.tail_trie:
    #         tailObj = cast(TailSlots, self.tail_trie[reversed_tail])
    #         return tailObj.get_head(head_index)
    #     return None

    def create_matching_attack_strings(self, epoch_string, required_attack_string=None):
        """
        This function for a given epoch will compute all the matching attack strings (without knowing the next epoch)

        Args:
            epoch_string: epoch outcome string without dot, typically length 32.
            required_attack_string: optional exact attack string (tail.head) that must
                be present after generation. If provided, the function does not return
                early unless this exact key already exists in the trie.
        """
        # first perform a fast check if adding any new attack string is needed
        ats,_ = find_longest_attack(epoch_string+".A", HonestAttack.alpha, self.filter)
        tail, _head = ats.rsplit(".", 1)
        #head_index = get_head_index(_head)
        reversed_tail=tail[::-1]
        if reversed_tail in self.tail_trie:
            if required_attack_string is None:
                log(4, f"Longest attack string {ats} already exists in utility function.")
                return
            req_obj, _ = self.get_node_attack_string(required_attack_string)
            if req_obj is not None:
                log(4, f"Required attack string {required_attack_string} already exists in utility function.")
                return
        # this is a bit slower function which return all the possible matching attack strings
        edges = decompose_longest_attack_dynamic(epoch_string, HonestAttack.alpha, self.filter)
        # We sort the attack strings by length of the tail first, then by the length of head (shorter tail first for same tail)
        edges.sort(key=lambda e: (len(e[0].rsplit('.', 1)[0]), len(e[0])))
        attack_string_map = {} # to avoid creating the same attack_string multiple times
        new_attack_strings = []
        #updated= False
        for parent_str, child_str in edges:
            # Resolve Child Object (Next Decision)
            child_node = None

            if not child_str.startswith('.') and not child_str == "":
                if child_str in attack_string_map:
                    child_node = attack_string_map[child_str]
                    if not isinstance(child_node, AttackString):
                        log(1, f"Warning! child_str {child_str} resolved to invalid type {type(child_node)}; skipping.")
                        child_node = None
                else :
                    # Try to look it up in utility 
                    child_node, _pf = self.get_attack_string(child_str)
                    if child_node is not None:
                        attack_string_map[child_str] = child_node
                            
                if child_node is None:
                    log(1,f"Warning! child_str {child_str} not found in attack_string_map during on-the-fly tail processing ({child_str}).")
                    continue
                    
            parent_attack_string_obj, _ = self.get_attack_string(parent_str, exact_matching=True)

            if parent_attack_string_obj is not None and parent_attack_string_obj not in new_attack_strings:
                log(4, f"On-the-fly tail already exists: {parent_str}, reuse.")
                attack_string_map[parent_str] = parent_attack_string_obj
            else:
                log(4, f"Adding on-the-fly tail: {parent_str} -> child: {child_str}")
                parent_tail_len=len(parent_str.rsplit('.', 1)[0])
                if child_str=='.' or child_str=='':
                    astring = parent_str
                    child_node = None
                else:                        
                    child_tail_len=len(child_str.rsplit('.', 1)[0])
                    astring = parent_str[:parent_tail_len-child_tail_len]

                attack_string_obj, _updated, _new_pf = self.save_edge_attack_string(astring, child_node)
                if attack_string_obj is None:
                    log(1, f"Warning! could not add on-the-fly tail {parent_str} (child: {child_str}).")
                    continue
                if attack_string_obj not in new_attack_strings:
                    new_attack_strings.append(attack_string_obj)
                
                attack_string_map[parent_str] = attack_string_obj

        if len(new_attack_strings)  > 0:
            # set next decisions
            for attack_string_obj in new_attack_strings:
                #attack_string_obj.compute_realizations()
                # we need to 
                attack_string_obj.compute_utility()
            #self.avg_utility()
        

    def get_longest_matching_attack_string_obj(self, two_epoch_string):
        """
        Find the TailSlots object whose tail is the longest tail of the
        given two-epoch string that also has a matching head_index entry.

        Parameters:
            two_epoch_string -- 'tail.head' formatted string

        Returns the TailSlots object, or None if no match is found.
        """
        tail, head = two_epoch_string.rsplit(".", 1)
        head_index = get_head_index(head)
        reversed_tail=tail[::-1]
        for _ in range(33):
            # longesttailObj = self.tail_trie.longest_prefix(reversed_tail).value
            try:
                longesttailObj = cast(TailSlots, self.tail_trie.longest_prefix(reversed_tail).value)
            except (KeyError, AttributeError):
                longesttailObj = None
            
            if longesttailObj is not None:
                attack_string=longesttailObj.matching_head(head_index)
                if attack_string is not None:
                    break
            
            if reversed_tail == "":
                break
            reversed_tail=reversed_tail[:-1]
        if longesttailObj is None:
            # Fall back to the honest tail to keep simulation alive when no specific match exists.
            return cast(TailSlots, self.tail_trie[""])
        return longesttailObj 
    
    def get_next_tail(self, tail_obj, head_index):
        """
        Follow the 'next decision' pointer from tail_obj at the given
        head_index and return the corresponding TailSlots object.

        Returns None if no next decision is set or the target tail is
        missing from the trie.
        """
        next_d = tail_obj.get_next(head_index)
        if next_d is not None:
            if hasattr(next_d, 'astring'):
                 astring = next_d.astring
                 tail = astring.split('.')[0] if '.' in astring else astring
                 # If honest node with '.', tail might be empty string? 
                 # HonestAttack astring is '.'
                 if astring == '.':
                     tail = ""
                 elif '.' in astring:
                     tail = astring.split('.')[0]
                 
                 # PATCH: Check if tail exists in dictionary
                 rev_key = tail[::-1]
                 if rev_key not in self.tail_trie:
                     # Log warning if needed, but return None to avoid crash
                     log(3, f"Warning: TailSlots {tail} (key {rev_key}) missing from self.tail_trie")
                     return None
                 return self.tail_trie[rev_key]
            
            # PATCH: Handle non-astring case key lookup too
            rev_key = next_d[::-1]
            if rev_key not in self.tail_trie:
                return None
            return self.tail_trie[rev_key]
        return None

    def items(self):
        """
        Iterate over all (attack_string_key, AttackString) pairs in the trie.
        Yields strings of the form 'tail.head'.
        """
        for rev_tail, val in self.tail_trie.items():
            tailObj = cast(TailSlots, val)
            for head, utility in tailObj.items():
                yield f"{rev_tail[::-1]}.{head}", utility

    def __len__(self):
        """
        retursn the number of attack strings
        """
        ret = 0
        for _, heades in self.tail_trie.items():
            ret += len(heades)
        return ret

    def save_edge_attack_string(self, astring, parent):
        """
        Save an attack string into the tree. Returns true if the 
        """
        if parent is not None and not isinstance(parent, AttackString):
            log(1, f"Warning! invalid parent type in save_edge_attack_string: {type(parent)}")
            return None, False, None
        if parent == None or parent.is_honest():
            attack_string = astring
        else:
            attack_string = astring + parent.get_attack_string()
        tail, head = attack_string.rsplit(".", 1)
        head_index = get_head_index(head)
        if not self.is_properattack_string(tail, head_index):
            return None, False, None
        reversed_tail = tail[::-1]
        if reversed_tail not in self.tail_trie:
            tailObj = TailSlots(tail)
            self.tail_trie[reversed_tail] = tailObj
            log(4, f"Create TailSlots object for {reversed_tail[::-1]}")
        else:
            tailObj = cast(TailSlots, self.tail_trie[reversed_tail])
            log(4, f"TailSlots object for {reversed_tail[::-1]} already exists")
        attack_string_objects=tailObj.get_matching_heades(head_index)
        updated = False
        attack_string_obj = None
        for attack_string_obj_ in attack_string_objects:
            log(4, f"A matching attack string {attack_string_obj_.attack_string} already exists in utility function.")
            if attack_string_obj_.attack_string == attack_string:
                attack_string_obj = attack_string_obj_
        if attack_string_obj is None:
            log(4, f"Adding {attack_string} as a new utility function.")
            nofork_attack_string = self.search_for_nofork(tail, head_index)
            log(4, f"Nofork attack string for {attack_string} is {nofork_attack_string.attack_string}.")
            tailObj.add_attack_string(astring, parent, nofork_attack_string, head_index)
            attack_string_obj = tailObj.get_head(head_index)
            updated = True
        for attack_string_obj_ in attack_string_objects:
            log(4, f"A matching attack string {attack_string_obj_.attack_string} already exists in utility function.")
            for edge in attack_string_obj_.attacks:
                if attack_string_obj is not None and attack_string_obj.extend_attack(edge):
                    updated = True
        return attack_string_obj, updated, tailObj



        # new_node.compute_realizations()
        # saved = tailObj.add_node(new_node, head_index)
        # new_node.compute_utility()

        # if saved:
        #     return new_node, tailObj
        # return None, None

    def search_for_nofork(self, tail, head_index):
        """
        Find the no-fork (baseline) AttackString for the given tail and
        head_index by walking up the trie toward shorter tails.

        Returns the honest AttackString if no matching entry is found.
        """
        reversed_tail = tail[::-1]
        reversed_tail = findLastAbeforeH(reversed_tail)
        while reversed_tail != "":
            try:
                nofork_tailObj = cast(TailSlots, self.tail_trie.longest_prefix(reversed_tail).value)
            except (KeyError, AttributeError):
                nofork_tailObj = None
            if nofork_tailObj != None:
                attack_string=nofork_tailObj.matching_head(head_index)
                if attack_string is not None:
                    return attack_string
                else:
                    log(3,f"TailSlots {nofork_tailObj.tail} has no head {head_index}")
                    reversed_tail = reversed_tail[:-1]
            else:
                log(4, f"There is no attack string saved for {tail}. Continue searching for nofork...")
                reversed_tail = reversed_tail[:-1]
        log(4,f"Honest attack is the nofork for {tail}.{str_head(head_index)}")
        return self.honest_attack
    
    def padHforHead(self, tail, head_index):
        """
        Return the number of 'H' characters that must be appended to the
        reversed tail to obtain a trie key that is head-free with respect
        to all existing keys.  Used by make_unique_heades.
        """
        pad = ""
        for h in range(32 - len(tail)):
            if not self.tail_trie.has_subtrie(tail[::-1] + pad):
                return len(pad)
            else:
                there_is_matching_attack_string = False
                for val in self.tail_trie.itervalues(tail[::-1] + pad):
                    tailObj = cast(TailSlots, val)
                    if tailObj.tail != tail:
                        if head_index != 0:
                            if tailObj.max_head() > head_index:
                                if tailObj.utility(head_index) != 0:
                                    there_is_matching_attack_string = True
                                    break
                            if tailObj.utility(0) != 0:
                                there_is_matching_attack_string = True
                                break
                        else:
                            there_is_matching_attack_string = True
                            break
                if not there_is_matching_attack_string:
                    return len(pad)
            pad += "H"
        return len(pad)


    def make_unique_heades(self, alphabet=['H', 'A']):
        """
        Generates unique, head-free heades for all existing tail_trie.
        Ensures each reversed tail (rev_tail) corresponds to a unique leaf in the Trie.
        """
        new_trie = pygtrie.CharTrie() 

        # Sort items by length DESCENDING to process longer strings first.
        # This prevents the issue where a short string ("A") blocks a longer one ("AH").
        # By inserting "AH" first, "A" can be padded to "AA" to avoid collision.
        sorted_items = sorted(self.tail_trie.items(), key=lambda x: len(x[0]), reverse=True)

        for rev_tail, val in sorted_items:
            tail_obj = cast(TailSlots, val)
            # BFS queue for finding a non-conflicting spot
            queue = [rev_tail]
            
            while queue:
                candidate = queue.pop(0)

                # Check 1: Is candidate a head of an EXISTING key?
                # has_subtrie returns True if the key is a head of at least one key in the trie.
                try:
                    is_head_of_existing = new_trie.has_subtrie(candidate)
                except AttributeError:
                    # Fallback if has_subtrie is missing in this version
                    is_head_of_existing = bool(new_trie.keys(prefix=candidate))

                # Check 2: Is an EXISTING key a head of candidate?
                # longest_prefix returns the longest prefix node (value, key)
                longest_pref_node = new_trie.longest_prefix(candidate)
                is_extension_of_existing = (longest_pref_node.key is not None)

                if not is_head_of_existing and not is_extension_of_existing:
                    # Found a valid spot
                    new_trie[candidate] = rev_tail
                    tail_obj.pad = candidate[len(rev_tail):]
                    log(3, f"Unique head found for {rev_tail[::-1]} -> {candidate[::-1]} (pad: {tail_obj.pad})")
                    break
                
                # Check 3: Exact match is covered by Check 1 & 2 (it is both head of existing and extension)
                
                # Conflict found: try extending with alphabet
                # We limit the depth implicitly by queue order, but theoretically could go deep.
                # Only extend if we are roughly within reasonable limits (though BFS guarantees shortest extension)
                if len(candidate) - len(rev_tail) < 10: # Safety break, though unlikely to hit with valid logic
                    for char in alphabet:
                        queue.append(candidate + char)
                else:
                    log(1, f"Warning: Could not find unique head for {rev_tail} within limits.")
                    break

    def avg_utility(self):
        """
        Compute the probability-weighted average continuation utility
        over all attack strings in the trie.

        Stores the result in self.avg_utility_ and returns it.
        Also calls make_unique_heades() to ensure probability weights
        are well-defined.
        """
        self.sum_prob = 0
        self.make_unique_heades()
        alpha=HonestAttack.alpha
        for rev_tail, val in self.tail_trie.items():
            if rev_tail=='':
                continue
            tailObj = cast(TailSlots, val)
            utility=tailObj.eas_utility()
            prob=tailObj.probability()
            self.avg_utility_ += prob * utility
            self.sum_prob += prob
        if self.sum_prob > 1.0:
            log(1, f"Warning! Avg utility sums up to {self.sum_prob} < 1.0")
        return self.avg_utility_


    def __repr__(self):
        """String representation of the utility function"""
        ret = f"Utility function for {HonestAttack.alpha} (heades {len(self.tail_trie)}): \n"
        for rev_tail, val in self.tail_trie.items():
            tailObj = cast(TailSlots, val)
            for head, utility in tailObj.items():
                head_index = get_head_index(head)
                attack_str_obj = tailObj.get_head(head_index)
                label = attack_str_obj.attack_string if attack_str_obj is not None else f"{rev_tail[::-1]}.{head}"
                ret += f"{label}, "
                # if debug:
                #     ret += "_" * self.padHforHead(
                #         tailObj.tail, get_head_index(head)
                #     )
                # ret += f"{rev_tail[::-1]}.{head} {utility:.3f}\n"  # {extraH_after_epoch}
        if self.avg_utility_ != None:
            ret += f"\n average utility:{self.avg_utility_}\n"
            ret += f" sum_prob:{self.sum_prob}\n"
        return ret

    def details(self):
        """Return a verbose multi-line string listing every attack string and its utility."""
        ret = f"Utility function for {HonestAttack.alpha} (heades {len(self.tail_trie)}): \n"
        for rev_tail, val in self.tail_trie.items():
            tailObj = cast(TailSlots, val)
            for head, utility in tailObj.items():
                ret += f"{rev_tail[::-1]}.{head} {utility}\n"
                ret += f"{tailObj}"
        if self.avg_utility_ != None:
            ret += f"\n average utility:{self.avg_utility_}\n"
        return ret

    def table(self, till):
        """
        Return a list of the `till` most probable attack strings, sorted by
        descending probability.  Each entry is a tuple:
            (attack_string_key, probability, utility, num_realizations)
        """
        attack_string_list = []
        for rev_tail, val in self.tail_trie.items():
            tailObj = cast(TailSlots, val)
            for head, utility in tailObj.items():
                head_index = get_head_index(head)
                attack_str_obj = tailObj.get_head(head_index)
                num_real = len(attack_str_obj.get_realizations()) if attack_str_obj is not None else 0
                attack_string_list.append((f"{rev_tail[::-1]}.{head}", attack_prob(rev_tail, head_index, HonestAttack.alpha), utility, num_real))
        sorted_attack_string_list = sorted(
                attack_string_list, key=lambda x: x[1], reverse=True
            )
        return sorted_attack_string_list[:till]
    
    def get_statistics(self):
        """Return a formatted string of all attack strings with their utilities and optional padding info."""
        ret = ""
        for rev_tail, val in self.tail_trie.items():
            tailObj = cast(TailSlots, val)
            for head, utility in tailObj.items():
                head_index = get_head_index(head)
                if debug:
                    ret += "_" * self.padHforHead(
                        tailObj.tail, head_index
                    )
                    # if extraH_after==1:
                    #    extraH_after_epoch='H'
                ret += f"{rev_tail[::-1]}.{head} {utility:.3f}\n"  # {extraH_after_epoch}
        if self.avg_utility_ != None:
            ret += f"\n average utility:{self.avg_utility_}\n"
            ret += f" sum_prob:{self.sum_prob}\n"
        return ret



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute and inspect the heuristic utility function for RANDAO attack strings.\n"
            "\n"
            "When called with -attack, generates all matching attack strings for the given\n"
            "epoch outcome and saves the resulting trie to a JSON file.\n"
            "When called without -attack, loads a pre-built attack tree and prints statistics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Compute utility for a specific attack string at alpha=35%:\n"
            "  python epoch_utility_function.py -attack AAHAHH -alpha 35 -log 3\n"
            "\n"
            "  # Load a pre-built attack tree and show statistics:\n"
            "  python epoch_utility_function.py -load attack_list.json -alpha 30 -log 2\n"
            "\n"
            "  # Use no-weak-forking filter:\n"
            "  python epoch_utility_function.py -attack AAAH -alpha 40 -filter no_weak_forking"
        ),
    )
    parser.add_argument(
        "-log",
        help="Verbosity level: 1=critical only, 2=warnings, 3=info, 4=debug, 5=trace (default: 6=all).",
        type=int,
        default=6,
    )
    parser.add_argument(
        "-plot", help="Plot the attack string hierarchy graph.", action="store_true"
    )
    parser.add_argument(
        "-load",
        help="Path to a pre-built attack tree JSON file (default: attack_list.json).",
        type=str,
        default="attack_list.json",
    )
    parser.add_argument(
        "-alpha",
        help="Adversary stake fraction as a percentage, e.g. 35 means alpha=0.35 (default: 30).",
        type=float,
        default=30,
    )
    parser.add_argument(
        "-attack",
        type=str,
        default="AAAAH",
        help=(
            "Epoch outcome string (A=adversary slot, H=honest slot).  The string is\n"
            "right-padded with H to 32 characters.  Example: AAHAHH"
        ),
    )
    parser.add_argument(
        "-filter",
        type=str,
        default="",
        help=(
            "Restrict the set of valid attack strings.  Options:\n"
            "  selfish_mixing   -- include selfish-mixing attacks\n"
            "  no_weak_forking  -- exclude weak forking variants\n"
            "  eas62            -- EAS-6/2 family only\n"
        ),
    )
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Arguments: {e} can not be parsed")
    set_debug(args.log)
    if '.' in args.attack:
            attack=args.attack.split('.')[0]
            log(1,f"Warning! epoch utility for tail slots is considered, which is {attack}")
    else:
        attack=args.attack 
    if attack!='':
        utility = EpochUtilityFunction(args.alpha/100.0)
        utility.create_matching_attack_strings(attack.rjust(32, "H"))
        save_policy.save_utility_function(utility, "realisation_string.json")
        exit(-1)

    debug = True
    res_log = {}
    res_log["parameters"] = vars(args)
    res_log["run"] = []
    honest = save_policy.load_attack_tree(args.load)
    attack_strings=EpochUtilityFunction(args.filter)
    #attack_strings.build(honest)
    res_alpha = []
    for i in range(33):
        res_alpha.append(
            {
                "count": i,
                "head": 0,
                "tail": 0,
                "headProb": 0,
                "tailProb": 0,
            }
        )
    for attack_string, prob in attack_strings.items():
        tail, head = attack_string.rsplit(".", 1)
        res_alpha[len(head)]["head"] += 1
        res_alpha[len(tail)]["tail"] += 1
        res_alpha[len(head)]["headProb"] += prob
        res_alpha[len(tail)]["tailProb"] += prob
    res_log["run"].append(
        {
            "alpha": HonestAttack.alpha,
            "attackStringNum": len(attack_strings),
            "histogram": res_alpha,
        }
    )
    if attack != "":
        epoch_string, next_epoch_string = create_two_epoch_string(attack)
        #log(1,f"{epoch_string} next epoch utility: {attack_strings.get_head_utility(epoch_string)}")
        log(1,f"{epoch_string} this epoch utility: {attack_strings.compute_utility([0], epoch_string,0)}")
    log(2, attack_strings)
    if HonestAttack.alpha == 0.2:
        save_policy.printComparison(attack_strings)
    res_log["run"].append(
        {
            "alpha": HonestAttack.alpha,
            "stat": res_alpha,
            #"export": attack_strings.avgUtility_old(export=True, plot=False),
        }
    )
    save_policy.save_utility_function(attack_strings, "realisation_string.json")
    save_policy.save_xml(res_log, "results")
