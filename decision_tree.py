"""
Per-epoch attack decision policy.

DecisionPoint holds all possible realizations for a given tail string and
picks the best forking action each epoch.  AttackDecisionTree wraps the
per-slot Monte Carlo loop, computes the running head utility, and applies
the forking threshold gate.

Key classes:
    DecisionPoint    -- stores realizations for a single tail; selects best fork
    AttackDecisionTree -- drives the slot-by-slot simulation loop

Threshold logic:
    When add_avg_utility=True (target mode) a myopic gate is applied so that
    w-amplified continuation utility does not force sacrifice-heavy forks.
"""

import json
import random
import math

from forking_string import ( count_head_Hs, 
                            create_two_epoch_string)
from logger import is_debug, log, set_debug
from realization import Realization
from tail_slots import toChain, attack_str, TailSlots
from save_policy import save_xml
from epoch_utility_function import EpochUtilityFunction

set_debug(2)

debug = False
debug2 = False


class DecisionPoint:
    """
    class for storing all nodes (attacks) for a given tail
    """

    utilityObj: object | None = None

    def __init__(self, realizations, tailObj):  # real_nodes
        self.tailObj = tailObj
        self.realizations = {}
        for real, realization in realizations.items():
            self.realizations[real] = Realization(realization)


    def best_realization(self, result_str, realizations):
        """
        result_str is the realization so far
        """
        best_utility = None
        self.best_real = None
        self.best_C = None
        self.best_N = None
        for real, realization in self.realizations.items():
            outcome_, utility = realization.outcome(result_str)
            exp_utility = realization.exp_utility
            if best_utility == None or exp_utility > best_utility:
                log(5,f"Remember this outcome as it has the best utility so far ({utility:.2f})")
                best_utility = exp_utility
                self.best_real = realization
            if len(real) > 0:
                if real[0] == "C":
                    if self.best_C == None or exp_utility > self.best_C.exp_utility:
                        self.best_C = realization
                else:
                    if self.best_N == None or exp_utility > self.best_N.exp_utility:
                        self.best_N = realization
        return self.best_real

    # def is_propose(self):
    #     return self.best_real.realization.node.is_propose(
    #         self.best_C.immediate_reward, self.best_N.immediate_reward
    #     )


class AttackDecisionTree:
    def __init__(
        self, two_epoch_string, alpha, head_utility, utility, fork_multiplier=1, add_avg_utility=True
    ):
        """
        The attack decision tree.
        """
        self.alpha = alpha
        self.utility = utility
        self.two_epoch_string = two_epoch_string
        self.add_avg_utility = add_avg_utility
        Realization.set_globals(alpha, head_utility, utility)
        self.fork_multiplier = fork_multiplier
        self.avg_head_utility = self.get_avg_head_utility(head_utility, alpha)
        log(3, f"Avg. head utility: {self.avg_head_utility}")
        tail, head = two_epoch_string.split(".")
        head_index = count_head_Hs(head) + 1
        self.tailObj = utility.get_longest_matching_attack_string_obj(two_epoch_string)
        if self.tailObj is None:
            log(1, f"Warning: no matching tail for {two_epoch_string}; falling back to honest tail")
            self.tailObj = utility.tail_trie[""]
        self.attack_string_obj = self.tailObj.matching_head(head_index)
        self.as_utility = self.attack_string_obj.utility_val if self.attack_string_obj else 0  
        self.eas_utility= self.tailObj.eas_utility()
        self.expected_percent = 100 * alpha + self.as_utility / 0.32
        self.attack_string=self.attack_string_obj.attack_string if self.attack_string_obj else ''
        log(3,f"simulate attack {self.attack_string} with utility {self.as_utility} (->{self.expected_percent:.1f})")
        self.public_chain = ""
        self.private_chain = ""
        self.public_chain_till_the_end_of_forking = None
        #public_votes = 0
        #private_votes = 0
        decision=self.attack_string_obj
        self.result_len=decision.pos
        self.forking_attacks=[]
        best_real = None
        
        dp=0
        for _ in range(self.result_len+2):
            log(3,f"--- Decision point ({dp}) {decision.get_attack_string()} ({self.attack_string}) the public chain is {self.public_chain}")
            dp+=1
            #fork_segment=decision.get_nofork_segment(head_index)
            best_real_before = best_real

            # first evaluate the new realisations at this decision point to continue the public_chain
            best_real = self.select_best_realization(decision, self.public_chain, best_real)

            pos = self.result_len - decision.pos 
            if best_real is not None:
                log(3,f"Best util at (slot {pos}) public chain {decision.get_attack_string()} ({self.public_chain}) is {best_real.exp_utility:.2f} private chain:{self.private_chain} {f"util was {best_real_before.exp_utility}" if best_real_before is not None else '(first attack)'}") 
            
            # the next decision point would be
            new_decision=decision.get_next_decision()
            if new_decision is None:
                new_decision=utility.honest_attack 
            ns=decision.pos-new_decision.pos

            # if the new realisation is better than what we had so far we start to build private chain
            if  best_real is None:
                log(3,"Do not start the attack yet")
                self.public_chain += 'C'*ns
            else:
                if ( best_real_before is None or best_real.exp_utility > best_real_before.exp_utility + 0.0001): 
                    if best_real_before is None:
                        log(3,f"At the first forking attack the best realisation is {best_real.exp_utility:.2f} private chain {best_real.realization_str}")
                    else:
                        log(3,f"At (slot {pos}) a better realisation is found {best_real.exp_utility:.2f} > {best_real_before.exp_utility:.2f} private chain {best_real.realization_str}")
                    # new better realisation is found
                    if self.private_chain != "":
                        log(3,f"Disregard the private chain ({best_real.exp_utility}). Score: {best_real.exp_utility:.2f}. Epoch outcomes : {best_real.epoch_outcome}")
                        if best_real_before is not None:
                            log(3,f"Dropped private chain was {best_real_before.realization_str} with utility {best_real_before.exp_utility:.2f}")
                        self.drop_private_branch()
                    
                    #log(3,f"Build the private chain {best_real.realization_str}")
                    if decision.is_forking_attack():
                        self.start_a_private_branch(best_real)
                        log(3,f"Decide to fork at {decision.attack_string} ({best_real.exp_utility:.2f} > treashold{-32*self.alpha}+{(utility.avg_utility_*self.fork_multiplier if EpochUtilityFunction.utility_multiplier > 0 and self.add_avg_utility else 0)}+{self.avg_head_utility} (best util: {best_real.exp_utility:.2f})) public chain: {self.public_chain} (fork:{self.public_chain_till_the_end_of_forking}) private_chain:{self.private_chain}")
                    else:
                        log(3,f"Perform selfish mixing at {decision.attack_string} public chain {self.public_chain}")
                        self.public_chain += best_real.get_realization_CN()
                        break
                else:
                    log(3,f"No better realization found at (slot {pos}) public chain {decision.get_attack_string()} ({self.public_chain}) ({best_real.exp_utility if best_real else 'N/A'}<={best_real_before.exp_utility if best_real_before else 'N/A'})")

                self.extend_public_chain_till(self.result_len-new_decision.pos)
                
            if decision.attack_string==".":
                break

            #if new_decision.attack_string not in two_epoch_string:
            #    log(1, f"Warning! Decision string {decision.attack_string} not in epoch string {two_epoch_string}")
            decision=new_decision
            if decision is None:
                log(3, f"Decision chain broken unexpectedly!")
                break
        self.nextnext_epoch, self.utility = Realization.get_outcome(self.public_chain, decision)
        self.realization_str = toChain(self.public_chain)
        self.action_plan = self.public_chain
        #
        reward = self.nextnext_epoch.count("A")
        sacrifice = self.count_sacrifice()
        percent = 100 * (reward - sacrifice) / 32
        if percent < self.expected_percent * 0.7:
            log(3, f"Very bad luck!!")

    def extend_public_chain_till(self, until):
        if self.public_chain_till_the_end_of_forking is not None and \
            len(self.public_chain_till_the_end_of_forking) > until and \
                until != self.result_len:
            self.public_chain=toChain(self.public_chain_till_the_end_of_forking[:until])
        else:
            if self.private_chain!="":
                log(3,f"Publish private chain {self.private_chain} to extend public chain to {until} slots")
                self.public_chain = toChain(self.private_chain)
                self.private_chain = ""
                self.public_chain_till_the_end_of_forking = None
            elif len(self.public_chain) != self.result_len:
                log(3,f"Warning! Continue building public chain ({self.public_chain }) until {until} slots without public_chain_till_the_end_of_forking ")
            else:
                log(3,f"Public chain is already complete ({self.public_chain })")

    def start_a_private_branch(self, best_real):
        self.forking_attacks.append(best_real)
        self.private_chain  = self.public_chain + best_real.get_realization_CN()
        self.public_chain_till_the_end_of_forking = self.public_chain + best_real.get_public_chain() 
        log(4,f"Start building a private chain {self.private_chain} the public chain till the end of forking would be {self.public_chain_till_the_end_of_forking}")
    
    def drop_private_branch(self):
        if len(self.forking_attacks) == 0:
            log(1,"Warning! trying to disregard private chain when there is none")
            return
        if self.public_chain_till_the_end_of_forking is None:
            log(1, "Warning! public_chain_till_the_end_of_forking is None while dropping private branch")
            self.private_chain = ""
            return
        self.public_chain = self.public_chain_till_the_end_of_forking[:len(self.public_chain)]
        self.forking_attacks.pop()
        self.private_chain = ""
        self.public_chain_till_the_end_of_forking = None     

    def select_best_realization(self, decision, public_chain, best_real):
        if not decision.is_forking_attack():
            for realization_str, realization_obj in decision.realization_plan.items():
                realization=Realization(realization_obj.get_action_plan())
                _, exp_utility = realization.outcome(public_chain, decision, realization_obj.value_r)
                if best_real is None or exp_utility > best_real.exp_utility:
                    # log(5,f"Remember this outcome as it has the best utility so far ({exp_utility:.2f})")
                    best_real = realization            
        else:
            best_exp_utility_fork = 0
            for realization_str, realization_obj in decision.realization_plan.items():
                realization=Realization(realization_obj.get_action_plan())
                _, exp_utility = realization.outcome(public_chain, decision, realization_obj.value_r)

                exp_utility_fork = realization_obj.get_treashold() - 32 * self.alpha - realization.sacrifice
                exp_utility_fork_base=exp_utility_fork
                if EpochUtilityFunction.utility_multiplier > 0 and self.add_avg_utility:
                    exp_utility_fork += self.utility.avg_utility_ * self.fork_multiplier * EpochUtilityFunction.utility_multiplier
    
                #exp_utility_fork += self.avg_head_utility
                log(4,f"For {realization_str} util:{exp_utility:.2f} the threshold is {exp_utility_fork:.2f}=> head:{self.avg_head_utility:.2f} base:{exp_utility_fork_base:.2f} eas:{self.utility.avg_utility_ * self.fork_multiplier}")

                # Use myopic threshold (immediate reward vs continuous_t) when add_avg_utility is on,
                # to prevent over-sacrificing when as_utility >> avg_utility_ in high-w (target) mode.
                # Selection among passing forks still uses full exp_utility (continuation-adjusted).
                if self.add_avg_utility:
                    # myopic: exp_immediate_reward > continuous_t  (equiv: exp_immediate_reward - sacrifice - alpha*32 > exp_utility_fork_base)
                    pass_threashold = (realization.exp_immediate_reward - realization.sacrifice - 32 * self.alpha) > exp_utility_fork_base + 0.000001
                else:
                    pass_threashold = exp_utility > exp_utility_fork + 0.000001
                    
                if ((best_real is None or exp_utility > best_real.exp_utility) and 
                        pass_threashold):
                    log(5,f"Remember this outcome as it has the best utility so far ({exp_utility:.2f}) larger than the threshold: {exp_utility_fork}")
                    best_real = realization
                    best_exp_utility_fork = exp_utility_fork
                else:
                    log(5,f"This outcome is not good enough ({exp_utility:.2f}) while the threshold is : {exp_utility_fork}")
        return best_real
    
    def get_avg_head_utility(self, head_utility, alpha):
        # find the highest index with non zero value
        max_index = 0
        for i, pref_utility in enumerate(head_utility):
            if pref_utility is not None and pref_utility != 0:
                max_index = i
        
        avg_head_utility = 0
        prop_sum = 0
        log(4, f"Calculate avg utility with alpha={alpha}, max_index={max_index}, base_util={head_utility[0]}")
        for i in range(1, max_index + 1):
            prob = alpha * pow(1 - alpha, i - 1)
            prop_sum += prob
            val = head_utility[i]
            if val is None or val == 0:
                val = head_utility[0]
            avg_head_utility += prob * val
            log(4, f" i={i}: prob={prob:.4f}, val={val:.4f}, cumul_util={avg_head_utility:.4f}")
        
        avg_head_utility += (1 - prop_sum) * head_utility[0]
        log(4, f" Remainder prob={1 - prop_sum:.4f}, val={head_utility[0]:.4f}, final={avg_head_utility:.4f}")
        return avg_head_utility
    
    def count_sacrifice(self):
        """
        used for debugging
        """
        realization_CN = toChain(self.realization_str)
        pos = 32 - len(realization_CN)
        ret = 0
        for i, block in enumerate(realization_CN):
            if block == "N" and self.two_epoch_string[pos + i] == "A":
                ret += 1
        return ret

    def count_forked(self):
        return self.action_plan.count('F')



    def count_votes(self, ns, pos, two_epoch_string, private_chain, alpha, decision, best_real, best_real_before):
        """
        Estimate public/private vote weights while a private chain is being built.
        Returns a tuple: (public_votes, private_votes).
        """
        public_votes = 0.0
        private_votes = 0.0

        if 'P' not in private_chain:
            return public_votes, private_votes

        segment_start = max(0, 32 - pos)
        segment_end = min(len(two_epoch_string), segment_start + max(ns, 0))
        attack_string_segment = two_epoch_string[segment_start:segment_end]
        last_h = attack_string_segment.rfind('H')

        if last_h == -1:
            return public_votes, private_votes

        published_rel = private_chain[last_h:].find('O')
        best_before_util = best_real_before.exp_utility if best_real_before is not None else -1
        best_util = best_real.exp_utility if best_real is not None else -1

        if published_rel != -1:
            next_published_a_block = last_h + published_rel
            public_votes += next_published_a_block * (1 - alpha)
            private_votes += next_published_a_block * alpha
            if public_votes >= private_votes + 0.4:
                log(1, f"warning! public votes exceeded private votes during private chain publication {public_votes}>={private_votes}+0.4")
                log(1, f"Publish private chain {private_chain} {decision.get_attack_string()} {two_epoch_string} ({best_util:.2f}<={best_before_util:.2f})")
        else:
            public_votes += ns * (1 - alpha)
            private_votes += ns * alpha
            if private_votes >= public_votes + 0.4:
                next_epoch_a = two_epoch_string[33:].find('A')
                if next_epoch_a != -1:
                    next_published_a_block = ns + next_epoch_a
                    public_votes += next_published_a_block * (1 - alpha)
                    private_votes += next_published_a_block * alpha
                    if public_votes >= private_votes + 0.4:
                        log(1, f"warning! through epoch boundary private votes exceeded public votes during private chain publication {private_votes}>={public_votes}+0.4")
                        log(1, f"Publish private chain {private_chain} {decision.get_attack_string()} {two_epoch_string} ({best_util:.2f}<={best_before_util:.2f})")

        return public_votes, private_votes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Demo of a single epoch to manipulate the RANDAO"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
        default=4,
    )
    parser.add_argument("-alpha", help="Simulate only alpha", type=int, default=20)
    parser.add_argument("-attack", type=str, default=".", help="Run attack string")
    parser.add_argument(
        "-um",
        type=float,
        help="utility parameter: The weight of an A slot in the utility function (utility_multiplier)",
        default=1,
    )
    parser.add_argument("-load", help="load attack strings", 
                        type=str, default="saved_utility.json")
    

    args = parser.parse_args()
    set_debug(2)

    alpha=args.alpha*0.01
    attack_strings=EpochUtilityFunction(alpha)
    #attack_strings.load(args.load)
    EpochUtilityFunction.utility_multiplier = args.um
    DecisionPoint.utilityObj = attack_strings
    set_debug(args.log)   
    #tail, head = args.attack.split(".")
    epoch_string, next_epoch_string = create_two_epoch_string(args.attack)
    head_utility = [0]
    adt = AttackDecisionTree(
        epoch_string + "." + next_epoch_string,
        alpha,
        head_utility,
        attack_strings,
        EpochUtilityFunction.utility_multiplier
    )
    nextnext_epoch_string = adt.nextnext_epoch
    realization = adt.realization_str
    attack_string = adt.attack_string
    log(3, f"After the attack the epoch string is {nextnext_epoch_string}")
    reward_ = nextnext_epoch_string.count("A")
    sacrifice_ = adt.count_sacrifice()
    forked_ = adt.count_forked()
    disregard_ = adt.action_plan.upper().count("D")
    self_forked_ = adt.action_plan.upper().count("S")
    missed_ = adt.action_plan.upper().count("M")
    log(2,f"{(reward_-sacrifice_)/32:.2f} attack:{attack_string} ({adt.action_plan} {adt.expected_percent*0.01:.2f}) sacrifice {sacrifice_} to have {reward_} slots in e+2 ({realization} -> {nextnext_epoch_string}) RANDAO outcomes seen:{len(Realization.randao_outcomes)}")
    #attack_strings.save("test.json")