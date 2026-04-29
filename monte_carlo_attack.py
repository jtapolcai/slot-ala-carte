"""
Monte Carlo simulator of RANDAO biasability attacks on Ethereum.

Simulates an adversary with fraction alpha of stake executing a forking
attack across multiple 32-slot epochs.  The adversary uses a pre-computed
attack string database together with an on-the-fly heuristic utility function
(EpochUtilityFunction) to decide whether to fork at each slot.

Usage (examples):
    # Sweep all alpha values:
    python monte_carlo_attack_fly.py -fly -epoch 10000

    # Single alpha with fixed attack string:
    python monte_carlo_attack_fly.py -fly -alpha 35 -attack AHAH -epoch 5000

    # Target-mode (optimise for specific target slot):
    python monte_carlo_attack_fly.py -fly -target -alpha 25 -epoch 20000

Key flags:
    -alpha      adversary stake percentage (integer, default 35)
    -epoch      number of epochs to simulate (default 1000)
    -target     use target-slot utility weighting (w=100)
    -attack     run with a fixed attack string (no heuristic)
    -log        verbosity level 1..5 (default 3)
"""

import json
from math import ceil
from forking_string import (create_two_epoch_string,
                            generate_epoch_string)
from logger import is_debug, log, set_debug
from realization import Realization, toChain
from save_policy import save_xml, save_utility_function
from epoch_utility_function import EpochUtilityFunction
from decision_tree import DecisionPoint, AttackDecisionTree
from tail_slots import TailSlots, get_head_index, str_head
from forking_attack import ForkingAttack
#from find_attack_string import find_longest_attack, find_longest_attack_dynamic
from honest_attack import HonestAttack
from attack_string import AttackString
from utility_distr import avg_utility

set_debug(2)

debug = False
debug2 = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monte-Carlo simulator of RANDAO attacks"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
        default=2,
    )
    parser.add_argument("-test", help="Run test.", action="store_true")
    parser.add_argument("-auto", help="Automatic parameter setting.", action="store_true")
    parser.add_argument("-target", help="Optimize for target slot utility.", action="store_true")
    parser.add_argument("-alpha", help="Simulate only alpha", type=int, default=25)
    parser.add_argument("-alpha_min", help="The smallest alpha", type=int, default=20)
    parser.add_argument("-alpha_max", help="The smallest alpha", type=int, default=49)
    parser.add_argument("-alpha_step", help="The steps in alpha", type=int, default=5)
    parser.add_argument(
        "-min_prob",
        type=float,
        help="The minimum probability of an attack string",
        default="0.001",
    )
    parser.add_argument(
        "-epoch", help="The number of epochs to simulate.", type=int, default=10000
    )
    parser.add_argument("-attack", type=str, default="", help="Run attack string")
    parser.add_argument(
        "-nosim", action="store_true", help="Do not simulate attack string"
    )
    parser.add_argument(
        "-um",
        type=float,
        help="utility parameter: The weight of an A slot in the utility function (utility_multiplier)",
        default=1,
    )
    parser.add_argument(
        "-pw",
        type=float,
        help="utility parameter: The head weight. it changes the wight of the head should be less than 1.0-alpha",
        default=0,
    )
    parser.add_argument(
        "-fm",
        type=float,
        help="utility parameter: The fork multiplier. it changes the threashold for forking",
        default=1,
    )
    parser.add_argument(
        "-max_sacr",
        type=int,
        help="Maximum sacrifice limit for attack strings",
        default=None,
    )
    parser.add_argument(
        "-filter",
        type=str,
        default="",
        help="Filter attack string. Implemented options: selfish_mixing,eas62,no_weak_forking",
    )
    parser.add_argument("-load", help="load attack strings", type=str, default="saved_utility.json")
    parser.add_argument("-warmup", help="Measurement starts after this many slots.", type=int, default=6)
    
    args = parser.parse_args()
    set_debug(args.log)
    if args.pw:
        TailSlots.head_weight = args.pw
    EpochUtilityFunction.utility_multiplier = args.um
    if args.target:
        EpochUtilityFunction.utility_multiplier = 100
    if args.attack != "":
        EpochUtilityFunction.utility_multiplier = 0
        log(1,f"Running attack string {args.attack} thus we set utility multiplier {EpochUtilityFunction.utility_multiplier}")
    
    res_log = {}
    res_log["parameters"] = vars(args)
    res_log["run"] = []
    alpha_range = range(args.alpha_min, args.alpha_max, args.alpha_step)
    if args.alpha:
        alpha_range = [args.alpha]
    for aa in alpha_range:
        alpha = aa / 100.0
        HonestAttack.alpha = alpha
        utility = EpochUtilityFunction(alpha, args.filter)
        reference_utility = avg_utility(alpha)
        if reference_utility is None:
            log(1, f"Warning: Could not generate reference utility for alpha={alpha}")
            continue
        else:
            utility.avg_utility_ = (reference_utility-alpha)*32
            log(1,f"Utility is generated on the fly for alpha={alpha} avg utility: {utility.avg_utility_:.2f}")
            if args.filter =="" and args.target is False:
                log(1,f"As a refrence, in Table 2 the average number of slots was {utility.avg_utility_/0.32+100*alpha:.2f}%")
        HonestAttack.set_limit_sacrifice(args.max_sacr)
        set_debug(args.log)
        # monte-carlo simulation
        epoch_string = generate_epoch_string(alpha)
        next_epoch_string = generate_epoch_string(alpha)
        current_tail=None
        next_tail=None
        nextnext_tail=None
        prev_attack_utility = 0.0
        total_reward = 0.0
        total_sacrifice = 0.0
        total_forked = 0.0
        total_missed = 0.0
        total_selfforked = 0.0
        total_disregard = 0.0
        total_outcome_seen = 0.0
        total_chance = 0.0
        total_honest = 0.0
        total_expected_percent = 0.0
        res_alpha = []
        if args.attack != "":
            if not args.log:
                set_debug(4)
            EpochUtilityFunction.utility_multiplier = 0
        for i in range(33):
            res_alpha.append(
                {
                    "count": i,
                    "HSThead": 0,
                    "HSTtail": 0,
                    "HSTreward": 0,
                    "HSTsacrifice": 0,
                    "HSTforked": 0,
                    "utilityHistogram": 0,
                }
            )
        effective_slots = max(1, args.epoch - args.warmup)
        first=True
        # Progress bar if log level is 2
        use_progress = args.log == 2
        if use_progress:
            try:
                from tqdm import trange
                slot_iter = trange(args.epoch, desc="Simulating slots", ncols=80)
            except ImportError:
                log(2, "tqdm not installed, proceeding without progress bar.")
                slot_iter = range(args.epoch)
        else:
            slot_iter = range(args.epoch)

        for i in slot_iter:
            if nextnext_tail is None or args.attack != "":
                head_utility = [0]
            else:
                head_utility = nextnext_tail.get_head_utility()
            if args.attack != "":
                tail, head = args.attack.split(".")
                epoch_string, next_epoch_string = create_two_epoch_string(args.attack)
                if first:
                    utility.create_matching_attack_strings(epoch_string)
                    attack_string_obj, tail_obj=utility.get_attack_string(args.attack, False)
                    if attack_string_obj is not None:
                        attack_string_obj.compute_utility()
                        exp_util = attack_string_obj.utility_val
                        for i in range(33):
                            res_alpha[i]["utility"] = attack_string_obj.exp_distr[i]
                        head_utility = [0]
                        first = False
            else:
                utility.create_matching_attack_strings(epoch_string)
                log(3, f"head utility: {head_utility}")



            adt = AttackDecisionTree(
                epoch_string + "." + next_epoch_string,
                alpha,
                head_utility,
                utility,
                args.fm,
                add_avg_utility=(args.attack == "")
            )
            nextnext_epoch_string = adt.nextnext_epoch
            current_tail=next_tail
            next_tail=nextnext_tail
            nextnext_tail = adt.tailObj
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
            #if args.attack == "":
            
            epoch_string = next_epoch_string
            next_epoch_string = nextnext_epoch_string
            
            if i >= args.warmup:
                if args.warmup>0 and (i==args.warmup):
                    log(1,f"Measurement starts now after warmup of {args.warmup} slots, now simulate {args.epoch-args.warmup} slots.")
                    if not args.target:
                        utility.avg_utility()
                tail, head = attack_string.split(".")
                res_alpha[len(head)]["HSThead"] += 1 / effective_slots
                res_alpha[len(tail)]["HSTtail"] += 1 / effective_slots
                res_alpha[reward_]["HSTreward"] += 1 / effective_slots
                res_alpha[sacrifice_]["HSTsacrifice"] += 1 / effective_slots
                res_alpha[forked_]["HSTforked"] += 1 / effective_slots
                res_alpha[max(0, reward_ - sacrifice_)]["utilityHistogram"] += 1 / effective_slots
                total_reward += reward_
                total_sacrifice += sacrifice_
                total_forked += forked_
                total_selfforked += self_forked_
                total_missed += missed_
                total_disregard += disregard_
                total_expected_percent+=adt.expected_percent*0.01
                total_outcome_seen += len(Realization.randao_outcomes)
                if len(Realization.randao_outcomes)==1:
                    total_honest+=1
                total_chance += 1 - pow(1 - alpha, len(Realization.randao_outcomes))
        alpha_stat = {}
        alpha_stat["Aslots"] = total_reward / effective_slots
        alpha_stat["slotsGain"] = (total_reward - total_sacrifice) / effective_slots
        alpha_stat["profit"] = (
            total_reward - total_sacrifice
            ) / effective_slots - alpha * 100
        alpha_stat["sacrifice"] = total_sacrifice / effective_slots
        alpha_stat["forked"] = total_forked / effective_slots
        alpha_stat["selfforked"] = total_selfforked / effective_slots
        alpha_stat["missed"] = total_missed / effective_slots
        alpha_stat["disregard"] = total_disregard / effective_slots
        alpha_stat["avgNumbOutcomeSeen"] = total_outcome_seen / effective_slots
        alpha_stat["honest"] = total_honest / effective_slots
        alpha_stat["expPercent"] =  total_expected_percent / effective_slots
        avg_tail_length=0
        avg_head_length=0
        for i in range(33):
            avg_tail_length+=i*res_alpha[i]["HSTtail"]
            avg_head_length+=i*res_alpha[i]["HSThead"]
        alpha_stat["avgHeadLength"] = avg_head_length
        alpha_stat["avgTailLength"] = avg_tail_length
        chance = total_chance / effective_slots
        alpha_stat["targetSlot"] = chance
        res_alpha.append(alpha_stat)
        log(1,f"Results in {(total_reward-total_sacrifice)/(0.32*effective_slots):.4f}% slots alpha={alpha*100}% and after {effective_slots} measured slots (out of {args.epoch}) (avg. number of outcomes seen {total_outcome_seen/effective_slots:.2f} -> {100*chance:.2f}% for target slot) (expected percent:{(100*total_expected_percent / effective_slots):.4f})")
        if not args.target:
            utility.avg_utility()
        log(1,f"number of attack strings:{len(utility)} expected {100 * alpha + utility.avg_utility_ / 0.32:.2f}% (from {100*alpha:.0f}%)")

        if args.attack!="":
            log(1,f"For attack string {args.attack} the measured utility {(total_reward-total_sacrifice)/effective_slots-alpha*32:.2f} (avg. {(total_reward-total_sacrifice)/(0.32*effective_slots):.0f}%) while the estimated {exp_util:.2f} ({(100 * alpha + exp_util/0.32):.0f}%)")
        if args.target and chance>=0.9999:
            break
        res_log["run"].append({"alpha": alpha, "slots": res_alpha})
    save_xml(res_log, "results")
    save_utility_function(utility, "realisation_string.json")
    if args.attack != "" and args.epoch>=1000:
        import os
        os.system(
            "python xml_process.py results.xml -x count -y utilityHistogram utility -fig"
        )
        log(1,"[visualize]  python xml_process.py -x count -y utilityHistogram HSTreward -fig results.xml")
