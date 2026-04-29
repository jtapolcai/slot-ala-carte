"""
Realization outcome evaluation and caching for RANDAO attack simulation.

A Realization represents one possible slot-by-slot outcome for the adversary
(O=opted, F=forked, P=private, H=honest, M=missed, S=self-forked).  The
outcome() method evaluates the attack decision for each head of the
realization string, returning the expected utility and immediate reward.

Caching: outcomes are keyed on the full chain string via randao_outcomes so
identical subproblems are solved only once per epoch.

Class globals (set via Realization.set_globals):
    alpha          -- adversary fraction
    head_utility -- continuation utility list from the previous epoch
    utilityObj     -- EpochUtilityFunction used for continuation lookup
"""

import random

from epoch_utility_function import EpochUtilityFunction
from forking_string import generate_epoch_string, count_sacrificed
from tail_slots import toChain
from logger import is_debug, log, set_debug

    
set_debug(2)
debug = True

class Realization:
    """
    a realization during attack

        realization string uses
        O : opted - adversory published block
        F: forked - honest forked block
        P: private built by adversory
        H: honest published block
        M: Missed slot by adversory
        S: Self-forked block by adversory
    """

    randao_outcomes = {}
    alpha = 0
    head_utility = None
    utilityObj: EpochUtilityFunction
    

    @classmethod
    def set_globals(cls, alpha, head_utility, utilityObj):
        # set global variables
        cls.alpha = alpha
        cls.head_utility = head_utility
        cls.utilityObj = utilityObj
        cls.randao_outcomes = {}
        cls.length = None

    def __init__(self, realization):
        self.realization = realization.upper()
        self.value_r = 0

    def outcome(self, pre_realization, decision=None, value_r=0):
        """
        generates the outcome of the realizations

        the results are stored in:
            self.realization
            self.epoch_outcome
            self.as_utility
            self.immediate_reward
            self.exp_immediate_reward
            self.sacrifice
        """
        self.realization_str = pre_realization + self.realization
        if self.length is None:
            self.length = len(self.realization)
        if debug:
            if self.length != len(self.realization):
                log(1,f"Warning! realization string {pre_realization} {self.realization} should be {self.length} long")
        realization_CN = toChain(self.realization_str)
        self.sacrifice = count_sacrificed(self.realization_str)
        log(3,f"Realization {realization_CN} ({pre_realization} {self.realization}) has sacrifice {self.sacrifice}")
        # we need to check if the same chain string was not already generated
        if realization_CN not in self.randao_outcomes:
            self.epoch_outcome = generate_epoch_string(self.alpha)
            self.as_utility, self.immediate_reward = self.utilityObj.compute_utility(
                self.head_utility, self.epoch_outcome, self.sacrifice
            )
            self.randao_outcomes[realization_CN] = self

            if decision is not None and EpochUtilityFunction.utility_multiplier > 0:
                self.exp_immediate_reward = decision.get_exp_immediate_reward(self.immediate_reward, value_r)
            else:
                self.exp_immediate_reward = self.immediate_reward
            self.exp_utility = EpochUtilityFunction.utility_multiplier * self.as_utility + self.exp_immediate_reward - self.sacrifice - self.alpha * 32

            log(3,f"For {realization_CN} ({pre_realization} {self.realization}) the output is {self.epoch_outcome} ({self.epoch_outcome.count('A')} -> {self.exp_immediate_reward:.2f} As) AS utility: {self.as_utility:.2f} (sacrifice:{count_sacrificed(self.realization_str)} (public chain {count_sacrificed(pre_realization)}), exp: {self.alpha*32}) expected utility:{self.exp_utility:.2f}")
            return self.epoch_outcome, self.exp_utility
        else:
            real = self.randao_outcomes[realization_CN]
            if real.sacrifice - self.sacrifice != 0:
                log(3,f"{self.realization_str} and {real.realization_str} has the same realization with different sacrifice")
            self.as_utility = real.as_utility
            self.immediate_reward = real.immediate_reward
            if decision is not None and EpochUtilityFunction.utility_multiplier > 0:
                exp_immediate_reward = decision.get_exp_immediate_reward(self.immediate_reward, value_r)
            else:
                exp_immediate_reward = self.immediate_reward
            if hasattr(self, 'exp_immediate_reward') and self.exp_immediate_reward != exp_immediate_reward:
                log(3,f"Expected immediate reward is updated (exp {exp_immediate_reward:.2f} vs {real.exp_immediate_reward:.2f})")
                self.exp_utility = real.exp_utility - (self.exp_immediate_reward - exp_immediate_reward)
            else:
                self.exp_utility = real.exp_utility
            self.exp_immediate_reward = exp_immediate_reward
            self.epoch_outcome = real.epoch_outcome
            log(3,f"For {realization_CN} ({pre_realization} {self.realization}) the output is already known with utility: {self.exp_utility:.2f} (was {real.exp_utility:.2f})")
            return self.epoch_outcome, self.exp_utility

    def get_public_chain(self):
        """
        returns the public chain built from the realization string
        """ 
        return self.realization.replace("P", "D").replace("F", "H").replace("S", "K").replace("O", "M")

    def get_realization_CN(self):
        return toChain(self.realization)
    
    # @classmethod
    # def get_outcome(cls, realization_str):
    #     """
    #     mostly called for honest attack
    #     """
    #     realization_CN = toChain(realization_str)
    #     if realization_CN in cls.randao_outcomes:
    #         real = cls.randao_outcomes[realization_CN]
    #         if toChain(real.realization_str) == toChain(realization_str):
    #             return real, (real.epoch_outcome, real.exp_utility)
    #     #if realization_str != "":
    #     #    log(2,f"Warning! Realization.get_outcome is called with unkown {realization_str} ({realization_CN})")
    #     real = Realization(realization_str)
    #     return real, real.outcome(realization_str)


    @classmethod
    def get_outcome(cls, realization_str, decision=None, value_r=0):
        """
        mostly called for honest attack
        """
        realization_CN = toChain(realization_str)
        if realization_CN in cls.randao_outcomes:
            real = cls.randao_outcomes[realization_CN]
            if toChain(real.realization_str) == toChain(realization_str):
                return real.epoch_outcome, real.exp_utility
        if realization_str != "":
            log(2,f"Warning! Realization.get_outcome is called with unkown {realization_str} ({realization_CN})")
        real = Realization(realization_str)
        return real.outcome(realization_str, decision, value_r)

