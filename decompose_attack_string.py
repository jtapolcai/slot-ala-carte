"""
Decompose a 32-slot epoch string into valid attack sub-strings.

decompose_longest_attack_dynamic(epoch_curr, alpha, filter) returns a list of
(attack_string, edge_type, next_position) triples that cover all valid
selfish-mixing and forking attack strings starting within the epoch.  The
decomposition is used by EpochUtilityFunction to populate the trie on the fly.
"""

from forking_string import (count_head_Hs, find_last_H, find_last_A, find_first_A, create_two_epoch_string,
                            forking_string_types, generate_epoch_string,
                            is_attack_string_type, max_forking_string_length)
from logger import is_debug, log, set_debug

set_debug(5)
debug = False


# return all decompositions for a single epoch
def decompose_longest_attack_dynamic(epoch_curr: str, alpha: float, filter="") -> list[tuple[str, str]]:
    assert len(epoch_curr) == 32

    last_h=find_last_H(epoch_curr)
    is_attack_string: list[bool] = [False] * 33 # the smallest head index is added
    is_attack_string[32] = True  # "." is always attack string
    a_1 = 0
    x_a = 0
    x_h = 0
    edges = []
    
    edges_through_epoch_boundary = []

    # the selfish mixing attacks after last_H
    for slot in range(31, last_h, -1):
        is_attack_string[slot] = True
        edges.append((slot, 'SM', slot + 1))

    for slot, adv_char in reversed(list(enumerate(epoch_curr[:last_h]))):
        if adv_char == "H":  # attack string cannot start with "H"
            a_1 = 0
            continue
        a_1 += 1
        x_a = max_forking_string_length(a_1, alpha, "A")
        x_h = max_forking_string_length(a_1, alpha, "H")
        
        current_is_attack_string = False
        
        if is_attack_string[slot + 1]:  # S -> "A" + S
            current_is_attack_string = True
            edges.append((slot, 'SM',slot + 1 ))

        # In selfish_mixing mode we explicitly skip all forking transitions.
        if filter == "selfish_mixing":
            is_attack_string[slot] = current_is_attack_string
            continue

        weak_forking_possible_till_epoch_boundary = False
        if filter != "no_weak_forking" and x_h is not None:
            # weak forking
            max_forking_length=slot + x_h + 1
            for slot_after_forking in range(slot + a_1 + 1, min(33, max_forking_length)):
                if is_attack_string[slot_after_forking]:
                    current_is_attack_string = True
                    if slot_after_forking >= 32:
                        weak_forking_possible_till_epoch_boundary = True
                        edges_through_epoch_boundary.append((slot, 'WF', slot_after_forking))
                    else:
                        edges.append((slot, 'WF', slot_after_forking))
                        

        if not weak_forking_possible_till_epoch_boundary and x_a is not None:
            # normal forking
            max_forking_length=slot + x_a
            for arrival_slot in range(slot + a_1 + 1, min(33, max_forking_length)):
                if arrival_slot >=32:
                    current_is_attack_string = True
                    edges_through_epoch_boundary.append((slot, 'F', max_forking_length))
                elif epoch_curr[arrival_slot] == "A" and is_attack_string[arrival_slot+1]:
                    current_is_attack_string = True
                    edges.append((slot, 'F', arrival_slot + 1))
        else:
            log(4, f"DEBUG {epoch_curr} Slot {slot}: weak forking possible till epoch boundary, skipping strong forking check.")

        #if position_of_a is not None and slot + x_a[slot] - 1 >= 32 + position_of_a and not weak_forking_possible_till_epoch_boundary:  # forking after epoch boundary
        #    current_is_attack_string = True
            # Forking lands on 'A' at (33 + position_of_a). Valid next state starts after it.
        #    edges.append((slot, 'F', 33 + position_of_a, 33 + position_of_a))
        
        is_attack_string[slot] = current_is_attack_string

    # Convert edges from slot indices to string representations
    # ignore selfish mixing attacks
    # filter edges that go beyond epoch boundary unnecessarily
    # avoid duplicates
    str_edges = []
    seen_edges = set()
    attack_string={32:{"."}}
    for u, fork_type, v in edges_through_epoch_boundary:
        next_epoch=""
        if v>=33:
            next_epoch="H"*(v-33)+"A"
        s_u = epoch_curr[u:] +'.'+ next_epoch
        if not is_attack_string_type(s_u, filter):
            continue
        if u not in attack_string:
            attack_string[u] = {s_u}
            log(4,f"save:{u}->{s_u}")
        else:
            list_u=attack_string[u]
            if s_u not in list_u:
                list_u.add(s_u)
        s_v='.'
        log(4, f"DEBUG {epoch_curr} Edge: {u} --{fork_type}--> {v} (through epoch) : {s_u} -> {s_v}")
        if (s_u, s_v) not in seen_edges:
            str_edges.append((s_u, s_v))
            seen_edges.add((s_u, s_v))

    for u, fork_type, v in edges:
        if v in attack_string:
            for s_v in attack_string[v]:
                s_u = epoch_curr[u:v] + s_v
                if not is_attack_string_type(s_u, filter):
                    continue
                if u not in attack_string:
                    attack_string[u] = {s_u}
                    log(4,f"save:{u}->{s_u}")
                else:
                    list_u=attack_string[u]
                    if s_u not in list_u:
                        list_u.add(s_u)
                log(4, f"DEBUG {epoch_curr} Edge: {u} --{fork_type}--> {v} (through epoch) : {s_u} -> {s_v}")
                if (s_u, s_v) not in seen_edges:
                    str_edges.append((s_u, s_v))
                    seen_edges.add((s_u, s_v))
        else:
            # This can happen with restrictive filters (e.g. eas62/selfish_mixing)
            # when the target state is pruned by filter constraints.
            if filter is None or filter == "":
                log(1, f"Warning {epoch_curr}: pruned edge target slot index {v} for filter '{filter}'")
    return str_edges

def generate_tikz(edges, alpha):
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)

    def split_node(node):
        if "." in node:
            head, tail = node.split(".", 1)
            return head, tail
        return node, ""

    groups = {}
    for node in nodes:
        head, tail = split_node(node)
        if head not in groups:
            groups[head] = []
        groups[head].append((tail, node))

    # Columns are ordered by head length (longest first).
    # Variants that only differ in tail are stacked vertically.
    ordered_heades = sorted(groups.keys(), key=len, reverse=True)
    node_list = []
    def tail_priority(tail):
        if tail == "":
            return (0, 0, "")
        if tail == "A":
            return (1, 0, "")
        return (2, len(tail), tail)

    for head in ordered_heades:
        node_list.extend([node for _, node in sorted(groups[head], key=lambda x: tail_priority(x[0]))])

    node_map = {name: i for i, name in enumerate(node_list)}

    def node_id(name):
        return name.replace(".", "p")

    print(r"\begin{tikzpicture}[>=stealth, node distance=1.5cm]")
    print(f"  % Nodes alpha={alpha}%")
    prev_anchor = None
    for head in ordered_heades:
        variants = sorted(groups[head], key=lambda x: tail_priority(x[0]))
        for row, (_, name) in enumerate(variants):
            if prev_anchor is None and row == 0:
                position = ""
            elif row == 0:
                position = f", right of ={node_id(prev_anchor)}"
            else:
                position = f", above of ={node_id(variants[row - 1][1])}, node distance=10mm"

            label = name.replace("_", r"\_").replace("A", r"\AS").replace("H", r"\HS")
            print(f'  \\node[asnode, font=\\footnotesize{position}] ({node_id(name)}) {{${label}$}};')

        prev_anchor = variants[-1][1]
    print(r"  % Edges")
    for u, v in edges:
        if node_map[v] == node_map[u] + 1:
            print(f'  \\draw[->] ({node_id(u)}) -- ({node_id(v)});')
        else:
            print(f'  \\draw[->] ({node_id(u)}.north) to[bend left=10] ({node_id(v)}.north);')

    print(r"\end{tikzpicture}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decompose an epoch into attack strings"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
        default=5
    )
    parser.add_argument("-test", help="Run test.", action="store_true")
    parser.add_argument("-alpha", help="Simulate only alpha", type=float, default=20)
    parser.add_argument("-attack", type=str, default="AAAHHH", help="Run attack string")
    parser.add_argument("-latex", help="Output TikZ code to stdout", action="store_true")
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"Arguments: {e} can not be parsed")
    set_debug(args.log)
    if args.test:
        import unittest
        loader = unittest.TestLoader()
        start_dir = 'tests'
        suite = loader.discover(start_dir)

        runner = unittest.TextTestRunner()
        runner.run(suite)
        exit(0)
    if '.' in args.attack:
        attack=args.attack.split('.')[0]
        log(1,f"Warning! epoch utility for tail slots is considered, which is {attack}")
    else:
        attack=args.attack 
    edges = decompose_longest_attack_dynamic(attack.rjust(32, "H"), args.alpha/100.0)
    if args.latex:
        generate_tikz(edges, args.alpha)

