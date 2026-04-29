"""
Helper functions for constructing and analysing RANDAO attack strings.

An attack string is a sequence of slot characters:
    A -- adversary block (published or kept private)
    H -- honest block
    . -- epoch boundary separator
    F -- fork point

Key functions:
    generate_epoch_string()          -- sample a random slot sequence
    create_two_epoch_string()        -- create a string spanning two epochs
    max_forking_string_length()      -- longest profitable forking tail
    count_sacrificed() / count_dropped() -- sacrifice / dropped block counts
    get_head_index()               -- head-index encoding (0='', 1='A', ...)
    toChain()                        -- convert realization to chain identifier
"""
import math
import random
from typing import Literal

Pboost = 0.4
from itertools import combinations, product

from logger import is_debug, log, set_debug

set_debug(2)

def count_sacrificed(realization_string):
    return (
        realization_string.count("M")
        + realization_string.count("S")
        + realization_string.count("D")
    )

def count_forked(realization_string):
    return realization_string.count("F")

def toChain(realization_string):
    translation = {
        "O": "C", "F": "N", "P": "C", "H": "C",
        "M": "N", "S": "N", "D": "N", "K": "C"
    }
    return realization_string.upper().translate(str.maketrans(translation))

def generate_epoch_string(alpha, length=32):
    """
    Generate a random string of specified length where each character is
    chosen independently. 'A' is chosen with probability alpha,
    and 'H' is chosen with probability 1-alpha.

    Args:
        alpha (float): Probability of choosing 'A' (default is alpha global variable).
        length (int): Length of the string (default is 32).
    Returns:
        str: Randomly generated string of 'A' and 'H'.
    """
    return "".join(random.choices(["A", "H"], weights=[alpha, 1 - alpha], k=length))


def is_forking_attack_feasible(a1, x, alpha, last_slot, confident=False):
    global Pboost
    confident_value = 0
    if confident:
        confident_value = 1 - alpha
    log(4,f"a1:{a1},x:{x},alpha:{alpha},last_slot:{last_slot},confident_value:{confident_value}")
    if last_slot == "A":
        log(4,f"forking string:{(x-1) * alpha + Pboost} >= {(1-alpha) * (x - a1 - 1) + confident_value}")
        return (x - 1) * alpha + Pboost + 0.00001 >= (1 - alpha) * (
            x - a1 - 1
        ) + confident_value
    log(4,f"weak forking string:{x * alpha} >= {(1-alpha) * (x - a1) + Pboost + confident_value}")
    return x * alpha + 0.00001 >= (1 - alpha) * (x - a1) + confident_value
    # return x * alpha + 0.00001 >= (1-alpha) * (x - a1) + Pboost + confident_value


def max_forking_string_length(a1, alpha, last_slot: Literal["A", "H"], confident=False):
    global Pboost
    confident_value = 0
    if confident:
        confident_value = 1 - alpha
    if last_slot == "A":
        ret = (
            math.floor(
                0.0001 + (a1 * (1 - alpha) + Pboost - confident_value) / (1 - 2 * alpha)
            )
            + 1
        )
    else:
        # ret=math.floor(0.0001+ (a1* (1 - alpha) - Pboost - confident_value) /(1 - 2*alpha))
        ret = math.floor(
            0.0001 + (a1 * (1 - alpha) - confident_value) / (1 - 2 * alpha)
        )
    if ret >= 1:
        return ret
    log(1, f"Warning! There is no valid forking string length is {ret} for a1:{a1}, alpha:{alpha}, last_slot:{last_slot}, confident_value:{confident_value}")
    return 0


def immediate_reward(epoch_string):
    return epoch_string.count("A")


def count_head_As(attack_string):
    count = 0
    for char in attack_string:
        if char == "A":
            count += 1
        if char == "H":
            break
    return count

def find_last_A(attack_string):
    return attack_string.rfind('A')

def find_first_A(attack_string):
    return attack_string.find('A')

def find_last_H(attack_string):
    return attack_string.rfind('H')

def count_head_Hs(attack_string):
    count = 0
    for char in attack_string:
        if char == "H":
            count += 1
        if char == "A":
            break
    return count


def count_trailing_Hs(s):
    count = 0
    for char in reversed(s):
        if char == "H":
            count += 1
        else:
            break
    return count

def findLastAbeforeH(reversed_tail):
    return reversed_tail.rstrip("A").rstrip("H")

def findLastH(reversed_tail):
    return reversed_tail.rstrip("A")


def forking_parameters(attack_string):
    a = attack_string.count("A")
    h = attack_string.count("H")
    a1 = count_head_As(attack_string)
    return a, h, a1


forking_string_types = ["selfish_mixing", "eas62", "no_weak_forking", "all"]


def is_attack_string_type(attack_string, filter=None):
    if filter is None or filter == "":
        return True
    tail, head = attack_string.split(".")
    if head == "":
        head_index = 0
    else:
        head_index = count_head_Hs(head) + 1
    return is_attack_string_param_type(tail, head_index, filter)


def is_attack_string_param_type(tail, head_index, filter=None):
    if filter == "" or filter == "norec" or filter == "all":
        return True
    elif filter == "selfish_mixing":
        if "H" in tail or head_index != 0:
            return False
        else:
            return True
    elif filter == "eas62":
        if len(tail) >= 7 or head_index >= 3:
            return False
        a1 = count_head_As(tail)
        if a1 == len(tail):
            return True
        h = count_head_Hs(tail[a1:])
        if len(tail) == a1 + h and head_index == 0:
            return False
        if len(tail) > a1 + h and head_index != 0:
            return False
        if "H" in tail[a1 + h :]:
            return False
        return True
    elif filter == "no_weak_forking":
        if tail != "" and ("H" == tail[-1] and head_index == 0):
            return False
        return True


def forking_realization_parameters(attack_string, alpha):
    """
    returns:
    a: the number of A slots
    h: the number of H slots
    a1: the number of consecutivy A slots at the begining of the attack_string
    a2: the first self-forking A slot among the A slots
    a3: the last+1 self-forking A slot among the A slots
    y: the lenght of the realization string
    a_before_epoch: the number of A in the realization string
    can_be_missed: the number slots that can be missed by A
    """
    a = attack_string.count("A")
    h = attack_string.count("H")
    a1 = count_head_As(attack_string)
    over_epoch_boundary = False
    last_slot = attack_string[-1]
    if "." in attack_string:
        tail_attack_string, head_attack_string = attack_string.split(".")
        if len(head_attack_string) > 0:
            over_epoch_boundary = True
        else:
            last_slot = tail_attack_string[-1]
        a_before_epoch = tail_attack_string.count("A")
        y = len(tail_attack_string)
    else:
        a_before_epoch = a
        y = a + h
    a3 = a_before_epoch
    log(4, f"last_slot:{last_slot}, over_epoch_boundary:{over_epoch_boundary}")
    if last_slot == "A" and not over_epoch_boundary:
        a3 = a3 - 1
    if not is_forking_attack_feasible(a1, a + h, alpha, last_slot):
        log(4,f"not valid forking attack string {attack_string} a1:{a1} x:{a+h}, alpha:{alpha}, last_slot:{last_slot}")
        return a, h, a1, None, None, None, None, None
    can_be_missed = a_before_epoch - 1
    if last_slot == "A" and not over_epoch_boundary:
        can_be_missed -= 1
    if can_be_missed >= 0:
        if is_forking_attack_feasible(a1, a + h, alpha, last_slot, True):
            log(4, f"confident forking attack string {attack_string}")
            a2 = 1
        else:
            log(4, f"not confident forking attack string {attack_string}")
            a2 = a1
    else:
        a2 = a3
    if a3 > a2:
        return a, h, a1, a2, a3, y, a_before_epoch, can_be_missed
    else:
        return a, h, a1, None, None, y, a_before_epoch, can_be_missed


def generate_forking_realizations(attack_string, alpha, limit_sacrifice=None):
    a, h, a1, a2, a3, y, a_before_epoch, can_be_missed = forking_realization_parameters(
        attack_string, alpha
    )
    if can_be_missed == None or can_be_missed < 0:
        log(4,f"Warning! can_be_missed:{can_be_missed} for attack_string:{attack_string}, alpha:{alpha}")
        return []
    cartesian_product = ["".join(p) for p in product("OM", repeat=can_be_missed)]
    realizations = []
    if len(cartesian_product) == 0:
        realization_string = (
            attack_string[:a_before_epoch].replace("A", "P").replace("H", "F")
        )
        if limit_sacrifice is None or count_sacrificed(realization_string) <= limit_sacrifice:
            realizations.append(realization_string)
    else:
        for element in cartesian_product:
            realization_string = "P"
            a_count = 0
            m_s_d_count = 0
            for c in attack_string[1:y]:
                if c == "H":
                    realization_string += "F"
                if c == "A":
                    if a_count < can_be_missed:
                        status = element[a_count]
                        if (
                            a2 != None
                            and a3 != None
                            and a_count + 1 >= a2
                            and a_count + 1 < a3
                            and status == "M"
                        ):
                            status = "S"
                        realization_string += status
                        if status in ("M", "S", "D"):
                            m_s_d_count += 1
                            if limit_sacrifice is not None and m_s_d_count >= limit_sacrifice + 1:
                                break
                    else:
                        realization_string += "O"
                    a_count += 1
            else:
                if limit_sacrifice is None or m_s_d_count <= limit_sacrifice:
                    log(4, f"realization:{realization_string}")
                    realizations.append(realization_string)
    return realizations

def create_two_epoch_string(attack_string: str) -> tuple[str, str]:
    assert "." in attack_string, "the attack string must contain the epoch boundary ."
    tail, head = attack_string.split(".")
    return tail.rjust(32, "H"), head.ljust(32, "H")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the parameters of a forking string"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
    )
    parser.add_argument("-test", help="Run test.", action="store_true")
    parser.add_argument(
        "-gentest", help="Run epoch string generator test.", action="store_true"
    )
    parser.add_argument("-alpha", help="Simulate only alpha", type=float, default=0.2)
    parser.add_argument("-attack", type=str, default="", help="Run attack string")
    args = parser.parse_args()
    if not args.log:
        set_debug(2)
    else:
        set_debug(args.log)
    if args.attack != "":
        a, h, a1, a2, a3, y, a_before_epoch, can_be_missed = (
            forking_realization_parameters(args.attack, args.alpha)
        )
        if y != None:
            log(1,f"{args.attack} -> a:{a}, h:{h}, a1:{a1}, realization string length:{y}, A before epoch boundary:{a_before_epoch} self-forking A slots:[{a2},{a3}] number of slots that can be missed:{can_be_missed}")
            realisations = generate_forking_realizations(args.attack, args.alpha)
            for realisation_string in realisations:
                log(1, f"{realisation_string}")

    if args.test:
        set_debug(5)
        for alpha in [0.2, 0.33]:
            log(1, f"for alpha:{alpha} ")
            for attack_string in [
                "AH.A",
                "AAH.A",
                "AAAH.A",
                "AAAHA.",
                "AAAHAHA.",
                "AAAAAAHAH.A",
            ]:
                a, h, a1, a2, a3, y, a_before_epoch, can_be_missed = (
                    forking_realization_parameters(attack_string, alpha)
                )
                log(
                    1,
                    f"{attack_string} -> a:{a}, h:{h}, a1:{a1}, realization string length:{y}, A before epoch boundary:{a_before_epoch} then self-forking A slot:[{a2},{a3}] can_be_missed:{can_be_missed}",
                )
                realisations = generate_forking_realizations(attack_string, alpha)
                for realisation_string in realisations:
                    log(1, f"{realisation_string}")
        epoch_string = generate_epoch_string(alpha)
    if args.gentest:
        sum_a = 0
        sum_h = 0
        sum_a1 = 0
        tries = 10000
        for i in range(tries):
            outcome = generate_epoch_string(args.alpha)
            a, h, a1 = forking_parameters(outcome)
            sum_a += a
            sum_h += h
            sum_a1 += a1
        log(1, f"for alpha:{args.alpha} avg a:{sum_a/tries} (should be {32*args.alpha})")
        log(1,f"for alpha:{args.alpha} avg h:{sum_h/tries} (shoule be {32-32*args.alpha})")
        log(1, f"for alpha:{args.alpha} avg a1:{sum_a1/tries} ")



def attack_str(tail,head_index):
    return f"{tail}.{str_head(head_index)}"

def str_head(head_index):
    if head_index == 0:
        return ""
    else:
        return f"{'H'*(head_index-1)}A"
     
def get_head_index(head):
    if head == "":
        return 0
    else:
        return count_head_Hs(head) + 1
