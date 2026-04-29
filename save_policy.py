"""
Serialization helpers: save and load attack policy and utility function data.

Functions:
    save_xml()              -- write the policy tree to an XML file
    save_utility_function() -- serialize the EpochUtilityFunction trie to JSON
    load_utility_function() -- restore a previously saved trie from JSON
    print_table()           -- pretty-print an attack string table to stdout
"""
import json
from logger import is_debug, log, set_debug
import xml.dom.minidom
import xml.etree.ElementTree as ET
from honest_attack import KnownOutcome, HonestAttack
from forking_string import str_head, get_head_index

class LoadedNode:
    """Helper class to store loaded node data"""
    def __init__(self, utility=0.0, next_decision=None):
        self.utility = utility
        self.next_decision = next_decision
        self.realization_plan = {}

def _format_realization_value(outcome, fallback_next_decision, chain_key=None, base_realization_plan=None):
    # For chained plans (e.g. selfish-mixing node carrying next_real), the
    # effective threshold is stored on next_real, not on the local outcome.
    threshold = outcome.treashold
    if threshold == 0.0 and outcome.next_real is not None and hasattr(outcome.next_real, 'treashold'):
        threshold = outcome.next_real.treashold

    # If still unknown, fall back to the same realization chain in the
    # head_index=0 node of this tail (e.g. AAH.A -> AAH.).
    if (
        threshold == 0.0
        and chain_key is not None
        and base_realization_plan is not None
        and chain_key in base_realization_plan
    ):
        base_outcome = base_realization_plan[chain_key]
        if hasattr(base_outcome, 'treashold') and base_outcome.treashold > 0.0:
            threshold = base_outcome.treashold

    continuation = outcome.next_real if outcome.next_real else (fallback_next_decision if fallback_next_decision else '')
    return f"{outcome}>={threshold} ({continuation})"
        
def tail_to_dict(tail_obj):
    ret = {
        "tail": tail_obj.tail,
        "eas_utility": tail_obj.eas_utility(),
        "probability": tail_obj.probability(),
        "head_utilities": tail_obj.get_head_utility(),
        "heades": []
    }
    base_realization_plan = None
    if 0 in tail_obj.attack_strings and tail_obj.attack_strings[0] is not None and hasattr(tail_obj.attack_strings[0], 'realization_plan'):
        base_realization_plan = tail_obj.attack_strings[0].realization_plan

    for i, attack_string in tail_obj.attack_strings.items():
        if isinstance(attack_string, LoadedNode):
            utility = attack_string.utility
            next_decision = attack_string.next_decision
            atk_str = None
            attack_string_val = None
        else:
            utility = attack_string.utility_val
            next_decision = attack_string.no_fork.attack_string if attack_string.no_fork else None
            atk_str = attack_string.attack_string if hasattr(attack_string, 'attack_string') else None
            attack_string_val = atk_str

        realizations = {
            k: _format_realization_value(v, next_decision, k, base_realization_plan)
            for k, v in attack_string.realization_plan.items()
        }

        entry = {
            "attack_string": atk_str if atk_str is not None else None,
            "utility": utility,
            "next_decision": next_decision,
            "realizations": realizations
        }
        ret["heades"].append(entry)
    return ret

def tail_from_dict(data):
    from tail_slots import TailSlots
    
    s_str = data.get('tail', data.get('tail', ''))
    obj = TailSlots(s_str)
    obj.avg_utility = data["eas_utility"]
    
    if "heades" in data:
        for item in data["heades"]:
            p_str = item.get("head", "")
            i = get_head_index(p_str)
            
            obj.attack_strings[i] = LoadedNode(
                utility=item["utility"],
                next_decision=item["next_decision"]
            )
            
            if "realizations" in item:
                # Expecting dict { "CN": "PF>=10.0" }
                for k, v in item["realizations"].items():
                    try:
                        real, thresh = v.split(">=")
                        outcome = KnownOutcome(real)
                        outcome.treashold = float(thresh)
                        obj.attack_strings[i].realization_plan[k] = outcome
                    except ValueError:
                         pass

    return obj

def save_utility_function(utility_function, file_name="attack_list.json"):
    with open(file_name, "w") as file:
        data = {}
        data['filter'] = utility_function.filter
        data['alpha'] = HonestAttack.alpha
        data['avg_utility'] = utility_function.avg_utility_
        data['tail_trie'] = [tail_to_dict(tail) for _, tail in utility_function.tail_trie.items()] 
        json.dump(data, file, indent=4)
        log(1,f"Attacks saved {file_name}")

def load_utility_function(utility_function, file_name="attack_list.json"):
    from tail_slots import TailSlots
    with open(file_name, "r") as file:
        data = json.load(file)
        utility_function.filter = data['filter']
        HonestAttack.alpha = data['alpha']
        utility_function.avg_utility_ = data['avg_utility']
        for item in data['tail_trie']:
            tailObj = tail_from_dict(item)
            utility_function.tail_trie[tailObj.tail[::-1]] = tailObj

def save_tex(content, file_name):
    with open(file_name, "w", encoding='utf-8') as file:
        file.write(content)

def save_attack_tree(file_name="attack_list.json", node_list=None):
    from honest_attack import HonestAttack
    if node_list is None:
        node_list = HonestAttack.node_list
    with open(file_name, "w") as file_obj:
        json.dump([node.to_dict() for node in node_list], file_obj, indent=4)

def load_attack_tree(file_name="attack_list.json"):
    from honest_attack import HonestAttack
    from selfish_mixing_attack import SelfishMixingAttack
    from forking_attack import ForkingAttack
    
    with open(file_name, "r") as file:
        data = json.load(file)

    ret = None        
    for item in data:
        if item["type"] == "honest":
            ret = HonestAttack.from_dict(item)
        elif item["type"] == "sm":
            SelfishMixingAttack.from_dict(item)
        elif item["type"] == "forking":
            ForkingAttack.from_dict(item)
    for item in data:
        if item["type"] == "forking" and item.get("nofork") is not None:
            HonestAttack.node_list[item["id"]].set_nofork(HonestAttack.node_list[item["nofork"]])
    return ret


def dict_to_xml(parent, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            child = ET.SubElement(parent, key)
            dict_to_xml(child, value)
        elif isinstance(value, list):
            for item in value:
                item_element = ET.SubElement(parent, key)
                if isinstance(item, dict):
                    dict_to_xml(item_element, item)
                else:
                    item_element.text = str(item).replace('_','')
        else:
            child = ET.SubElement(parent, key)
            child.text = str(value).replace('_','')

def save_xml(res_dict, name):
    root = ET.Element("Results")
    dict_to_xml(root, res_dict)
    xml_str = ET.tostring(root, encoding="utf-8")
    dom = xml.dom.minidom.parseString(xml_str)
    pretty_xml_as_string = dom.toprettyxml()
    with open(name + ".xml", "w") as xml_file:
        xml_file.write(pretty_xml_as_string)
    print(f"Result saved to {name}.xml")


def print_table(table, alpha_range,table_row_number, no_utility=False):
    import save_policy
    from find_longest_attack_string import find_longest_attack
    from forking_string import create_two_epoch_string

    if table_row_number <= 0 or not table:
        return

    
    alpha_values = list(alpha_range)
    cols = min(len(alpha_values), len(table))
    content = "\\begin{tabular}{"
    if no_utility:
        content += f"c|" * cols
    else:
        content += f"c c c|" * cols
    content += "}\n\\toprule\n"
    row = ""
    for aa in alpha_values[:cols]:
        if row != "":
            row += "&"
        if no_utility:
            row += "$\\alpha=" + str(aa) + "\\%$ "
        else:
            row += "\\multicolumn{3}{c|}{$\\alpha=" + str(aa) + "\\%$} "
    content += f"{row}\\\\\n"
    content += "\\midrule\n"
    for j in range(table_row_number):
        row = ""
        for i in range(cols):
            if row != "":
                row += "&"

            if len(table[i]) > j:
                attack_raw = table[i][j][0]
                aa = alpha_values[i]

                epoch_string, next_epoch_string = create_two_epoch_string(attack_raw)
                attack_string_, _min_head = find_longest_attack(
                    f"{epoch_string}.{next_epoch_string}", aa * 0.01, "no_weak_forking"
                )
                marked = attack_string_ != attack_raw
                mark_begin = "\\textbf{" if marked else ""
                mark_end = "}" if marked else ""

                attack_string = (
                    attack_raw.replace('A', '\\AS')
                    .replace('H', '\\HS')
                    .replace('.', '\\epoch')
                )

                if no_utility:
                    suffix = "^*" if marked else ""
                    row += f"${attack_string}{suffix}$ "
                else:
                    utility_value = table[i][j][2] if len(table[i][j]) > 2 else 0
                    count_value = table[i][j][3] if len(table[i][j]) > 3 else ""
                    if utility_value is None:
                        utility_text = "-"
                    else:
                        utility_text = f"{utility_value:.1f}"
                    row += f"${attack_string}$ & {mark_begin}{utility_text}{mark_end} & {mark_begin}{count_value}{mark_end}"
            else:
                if no_utility:
                    row += " "
                else:
                    row += " & & "

        content += f"{row}\\\\\n"

    content += "\\bottomrule\n\\end{tabular}\n"
    save_policy.save_tex(content, 'popular_attack_string_table.tex')



def printComparison(attack_strings):
    mdp_averaged = {
        ".": 0,
        "A.": 0.952,
        "AA.": 1.762,
        "AAA.": 2.459,
        "AAAA.": 3.069,
        "AAAAA.": 3.611,
        "AHAAAA.": 3.621,
        "AHAAA.": 3.051,
        "AAHAAA.": 3.455,
        "AHAA.": 2.402,
        "AAHAA.": 2.861,
        "AAAHAA.": 3.288,
        "AHA.": 1.655,
        "AAHA.": 2.185,
        "AAAHA.": 2.682,
        "AAAAHA.": 3.176,
        "AHAAHA.": 2.537,
        "AH.A": 1.149,
        "AAH.A": 1.915,
        "AAAH.A": 2.553,
        "AAAAH.A": 3.123,
        "AHAAAH.A": 3.068,
        "AHAAH.A": 2.332,
        "AAHAAH.A": 2.658,
    }
    print("Our compared to exported MDP utility:")
    print("attack string: our vs exported MDP utility")
    for rev_tail, tailObj in attack_strings.tail_trie.items():
        for head, utility in tailObj.items():
            astring = f"{tailObj.tail}.{head}"
            if astring in mdp_averaged:
                print(f"{astring}: {utility:.3f} vs {mdp_averaged[astring]}")
            #else:
            #    print(f"{astring}: {utility:.3f} vs N/A")

