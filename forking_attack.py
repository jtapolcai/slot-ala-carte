"""
ForkingAttack: forking attack node.

Models the strategy where the adversary forks the chain at a chosen slot,
keeping its blocks private and revealing them only when a profitable
RANDAO outcome is observed.  Extends HonestAttack with forking-specific
realization generation and expected utility computation.
"""

import math
import save_policy
from honest_attack import *
from selfish_mixing_attack import *
from utility_distr import *
from forking_string import count_sacrificed, generate_forking_realizations
from logger import is_debug, log, set_debug
# from attack_string import AttackString

set_debug(2)
debug = True

class ForkingAttack(HonestAttack):    # global variables for memory efficiency
    def __init__(self, astring: str, parent):  
        self.astring = astring
        if parent is None:
            self.pos=len(astring.split('.')[0])
        else:
            self.pos=len(astring) + parent.pos
        self.parent = parent  # Reference to parent attack string object
        self.id=len(self.node_list)
        self.node_list.append(self)
        self.compute_realizations()

    def is_honest(self):
        return False
            
    def compute_realizations(self):
        # if HonestAttack.limit_sacrifice is not None:
        #     prev_sacr=0
        #     if self.parent is not None:
        #         prev_sacr=count_sacrificed(self.parent.attack_string)
        self.realizations = [r for r in generate_forking_realizations(self.astring, HonestAttack.alpha, HonestAttack.limit_sacrifice)]
        

    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "type": "forking",
            "id": self.id,
            "astring": self.astring,
            "parent": self.parent.attack_string_obj.attack_string if self.parent else '',
        }

    @classmethod
    def from_dict(cls,data):
        parent = None
        if "parent" in data and data["parent"] is not None:
            if len(HonestAttack.node_list) > data["parent"]:
                parent = HonestAttack.node_list[data["parent"]]
            else:
                 log(1,f'Warning! with order of nodes in HonestAttack.node_list {data["id"]} has parrent {data["parent"]}')
        # Fallback for old data or if multiple parents existed (deprecated)
        elif "parents" in data:
            for pid in data["parents"]:
                if len(HonestAttack.node_list) > pid:
                    parent = HonestAttack.node_list[pid]
                    break
   
        obj = ForkingAttack(data["astring"],parent)
            
        if obj.id != data["id"]:
            log(1,f"Warning loading tree id:{obj.id}!= {data['id']}")
        return obj




if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(
        description="Demo of forking attack strings"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
    )
    parser.add_argument("-alpha", help="Simulate only alpha", type=int, default=20)
    parser.add_argument("-load", help="Export utility", action="store_true")
    args = parser.parse_args()
    if not args.log:
        set_debug(2)
    else:
        set_debug(args.log)
    if args.load:
        honest2 = save_policy.load_attack_tree("attack_list.json")
        save_policy.save_attack_tree("attack_list2.json")
        
