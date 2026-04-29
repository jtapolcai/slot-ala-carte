"""
Find the longest valid attack string within a two-epoch slot sequence.

find_longest_attack(two_epoch_string, alpha, filter) traverses the epoch
boundary from left to right and returns the longest attack string whose
probability exceeds the minimum threshold.  Used during trie population and
offline attack string database construction.
"""

from forking_string import (count_head_Hs, find_last_H, find_last_A, find_first_A, create_two_epoch_string,
                            forking_string_types, generate_epoch_string,
                            is_attack_string_type, max_forking_string_length)
from logger import is_debug, log, set_debug
from save_policy import save_xml, print_table
from decompose_attack_string import generate_tikz

set_debug(4)
debug = False


def find_longest_attack(two_epoch_string, alpha, filter=""):
    """
    Finds the longest attack string by traversing a two-epoch string from left to right.

    Parameters:
    - two_epoch_string (list): A list representing the epoch string with slots.
    - alpha (float): A threshold parameter that may influence decision-making.

    Returns:
    - longest_attack (list): The longest attack string found.
    """

    start = 0
    max_x = 0
    max_xh = 0
    a_1 = 0
    x_h = 0
    x_a = 0

    attack_start = 0
    attack_end_min = 32
    attack_end_max = 0
    # Traverse from left to right
    weak_forking_possible_till_epoch_boundary = False
    for i in range(33):
        if two_epoch_string[i] == "A":
            a_1 += 1
            if (
                a_1 >= 1 and two_epoch_string[i + 1] == "H"
            ):  # or two_epoch_string[i+1] == "."):
                # The actual start of the current A-run (independent of previous forkings).
                current_start = i - a_1 + 1
                # Chaining logic: when the current A lands exactly at max_x (terminal of
                # previous forking), either chain (a_1>1) or restart fresh (a_1==1).
                if i == max_x and (i-a_1-1>max_xh or filter == "no_weak_forking") and i != 31:
                    if a_1 > 1:
                        A_must_be_published = 1
                    else:
                        # Single A at the terminal slot: cannot chain (nothing to subtract).
                        # Restart fresh from this A position.
                        A_must_be_published = 0
                        start = current_start
                else:
                    A_must_be_published = 0
                # Compute maximum end position for regular forking
                x_a = (
                    i
                    + max_forking_string_length(a_1 - A_must_be_published, alpha, "A")
                    - a_1
                )
                if filter != "no_weak_forking":
                    # Compute maximum end position for weak forking
                    xH_max = (
                        max_forking_string_length(a_1 - A_must_be_published, alpha, "H")
                        - a_1
                    )
                    if xH_max >= 1:
                        x_h = i + xH_max
                        if x_h >= 31:  # If x_h reaches tail slot or next epoch
                            if attack_start <= 32 - current_start:
                                start = current_start
                                attack_start = 32 - start
                                attack_end_min = 0
                                weak_forking_possible_till_epoch_boundary = True
                            # return two_epoch_string[start:33]
                        max_x = max(max_x, x_h)
                        max_xh = max(max_xh, x_h)
                    else:
                        x_h = i + 1
                else:
                    x_h = i + 1
                if x_h >= 32:
                    x_h += 1
                if x_a >= 32:
                    x_a += 1
                last_A=find_last_A(two_epoch_string[x_h : x_a + 1])
                if last_A!=-1:
                    pos = x_h + last_A
                    if pos == 31 or (pos>=31 and two_epoch_string[31] == "A"):
                        if attack_start <= 32 - current_start:
                            start = current_start
                            attack_start = 32 - start
                            attack_end_min = 0
                            # attack string could terminate at epoch boundary
                    if pos >= 33:
                        if attack_start <= 32 - current_start:
                            start = current_start
                            attack_start = 32 - start
                        first_A=find_first_A(two_epoch_string[33 : x_a + 1])
                        if first_A>=0 and weak_forking_possible_till_epoch_boundary is False and attack_end_min>first_A:
                            attack_end_min = first_A
                        if attack_end_max < pos - 32:
                            attack_end_max = pos - 32
                        # return two_epoch_string[start:pos+1] # epoch boundary
                    max_x = max(max_x, pos)
        elif two_epoch_string[i] == ".":
            attack_start = 32 - start
            attack_end_min = 0
        else:
            a_1 = 0
            if i > max_x:
                start = i + 1
    # Canonical head form is H*A (or empty); trim any trailing As after the first one.
    if attack_end_max > 0:
        head = two_epoch_string[33 : attack_end_max + 33]
        first_A_in_head = find_first_A(head)
        if first_A_in_head >= 0:
            attack_end_max = first_A_in_head + 1
    return two_epoch_string[32 - attack_start : attack_end_max + 33], attack_end_min


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a table of the most common RANDAO attack strings"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
        default=3
    )
    #    parser.add_argument('-test', help='Run test.', action='store_true')
    parser.add_argument("-test", help="Run test.", action="store_true")
    #    parser.add_argument('-plot', help='Plot attack strign hierarchy graph', action='store_true')
    #    parser.add_argument('-export', help='Export utility', action='store_true')
    parser.add_argument("-alpha", help="Simulate only alpha", type=float, default=None)
    parser.add_argument("-alpha_step", help="The steps in alpha", type=int, default=5)
    parser.add_argument("-alpha_min", help="The smallest alpha", type=int, default=20)
    #    parser.add_argument('-min_prob', type=float, help='The minimum probability of an attack string', default="0.001")
    parser.add_argument("-attack", type=str, default="", help="Run attack string")
    parser.add_argument(
        "-epoch", help="The number of epochs to simulate.", type=int, default=10000
    )
    parser.add_argument("-table", help="Export a table of X attack strings with highest probability", type=int, default=0)
    parser.add_argument("-utility", help="Include utility values in the table", action="store_true")
    parser.add_argument("-latex", help="Output TikZ code to stdout", action="store_true")
    
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Arguments: {e} can not be parsed")
    # args.test=True
    set_debug(args.log)
    if args.test:
        import unittest
        loader = unittest.TestLoader()
        start_dir = 'tests'
        suite = loader.discover(start_dir)

        runner = unittest.TextTestRunner()
        runner.run(suite)
        exit(0)
    debug = True
    res_log = {}
    res_log["run"] = []
    table=[]
    alpha_range = range(args.alpha_min, 49, args.alpha_step)
    if args.alpha:
        alpha_range = [args.alpha]
    for aa in alpha_range:
        alpha = aa / 100.0
        res_alpha = []
        if args.attack != "":
            epoch_string, next_epoch_string = create_two_epoch_string(args.attack)
            two_epoch_string = epoch_string + "." + next_epoch_string
            attack_string, min_head = find_longest_attack(two_epoch_string, alpha)
            # attack_string, min_head = find_longest_attack(two_epoch_string, alpha,"no_weak_forking")
            if args.latex:
                log(1, "-latex is currently unsupported in find_longest_attack_string.py")
            log(1, f"attack string: {attack_string}")
        else:
            epoch_string = generate_epoch_string(alpha)
            next_epoch_string = generate_epoch_string(alpha)
            attack_strings = {}
            types = []
            for type_name in forking_string_types:
                type = {}
                type["tprob"] = 0
                type["tcount"] = 0
                type["name"] = type_name.replace("_", " ")
                types.append(type)
            attack_string_list = []
            for i in range(33):
                attack_string_list.append(
                    {
                        "pst_prob": 0,
                        "pst_count": 0,
                        "prf_prob": 0,
                        "prf_count": 0,
                        "prf_min_prob": 0,
                        "prf_min_count": 0,
                        "length": i,
                    }
                )
            for i in range(args.epoch):
                two_epoch_string = epoch_string + "." + next_epoch_string
                if True:
                    attack_string, min_head = find_longest_attack(two_epoch_string, alpha)
                    attack_stringH, min_headH = find_longest_attack(
                        two_epoch_string, alpha, "no_weak_forking"
                    )
                else:
                    attack_string = find_longest_attack_dynamic(two_epoch_string, alpha)
                    attack_stringH = find_longest_attack_dynamic(
                        two_epoch_string, alpha, "no_weak_forking"
                    )
                    min_head=0
                tail, head = attack_string.split(".")
                if attack_string not in attack_strings:
                    attack_strings[attack_string] = 0
                    for i, type in enumerate(forking_string_types):
                        if type != "no_weak_forking":
                            if is_attack_string_type(attack_string, type):
                                types[i]["tcount"] += 1
                        else:
                            if attack_string == attack_stringH:
                                types[i]["tcount"] += 1
                    attack_string_list[len(tail)]["pst_count"] += 1
                    attack_string_list[len(head)]["prf_count"] += 1
                    attack_string_list[min_head]["prf_min_count"] += 1
                attack_strings[attack_string] += 1.0 / args.epoch
                for i, type in enumerate(forking_string_types):
                    if is_attack_string_type(attack_string, type):
                        types[i]["tprob"] += 1 / args.epoch
                    else:
                        if type == "eas62":
                            log(3, f"{attack_string} is not eas62")
                attack_string_list[len(tail)]["pst_prob"] += 1 / args.epoch
                attack_string_list[len(head)]["prf_prob"] += 1 / args.epoch
                attack_string_list[min_head]["prf_min_prob"] += 1 / args.epoch
                log(
                    3,
                    f"{i}: {two_epoch_string} attack string is {attack_string} (no weak forking: {attack_stringH})",
                )
                epoch_string = next_epoch_string
                next_epoch_string = generate_epoch_string(alpha)
            # attack_string_list=[]
            # if len(attack_strings)<1:
            #    for attack_string,prob in attack_strings.items():
            #        tail,head=attack_string.split('.')
            res_log["run"].append(
                {
                    "alpha": alpha,
                    "stat": res_alpha,
                    "export": attack_string_list,
                    "type": types,
                    "numb": len(attack_string),
                }
            )
            log(2,f"for alpha={alpha} there are {len(attack_strings)} attack strings after {args.epoch} slots")
            attack_string_list=[]
            for attack_string, prob in attack_strings.items():
                attack_string_list.append((attack_string, prob, 0,0))
            sorted_attack_string_list = sorted(
                    attack_string_list, key=lambda x: x[1], reverse=True
                )
            if args.table>0:
                if args.utility:
                    from epoch_utility_function import EpochUtilityFunction
                    enriched_list = []
                    for atk_str, prob, _, _ in sorted_attack_string_list[:args.table]:
                        # Use a fresh utility trie per attack string. Reusing one shared
                        # object across entries can leak state between lookups and make
                        # table values depend on processing order.
                        utility_func = EpochUtilityFunction(alpha)
                        if "." in atk_str:
                            attack = atk_str.split(".")[0]
                        else:
                            attack = atk_str
                        utility_func.create_matching_attack_strings(
                            attack.rjust(32, "H"), required_attack_string=atk_str
                        )

                        # 1) exact lookup for the full attack string (e.g. AHHH.A)
                        atk_obj, _ = utility_func.get_node_attack_string(atk_str)

                        # 2) fallback: tail-only lookup (e.g. AHHH.) — covers empty head
                        if atk_obj is None:
                            tail_only = attack + "."
                            atk_obj, _ = utility_func.get_node_attack_string(tail_only)

                        # 3) best-match fallback: same tail, closest compatible head
                        #    e.g. AHHH.A -> AHHH.HA if only that head index exists
                        if atk_obj is None:
                            atk_obj, _ = utility_func.get_attack_string(
                                atk_str if "." in atk_str else attack + ".",
                                exact_matching=False
                            )

                        utility_val = atk_obj.utility if atk_obj is not None else None
                        num_real = len(atk_obj.get_realizations()) if atk_obj is not None else ""
                        enriched_list.append((atk_str, prob, utility_val, num_real))
                    table.append(enriched_list)
                else:
                    table.append(sorted_attack_string_list[:args.table])
    print_table(table, alpha_range,args.table,not args.utility)
    save_xml(res_log, "results")
