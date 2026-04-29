"""
SelfishMixingAttack: selfish-mixing (SM) attack node.

Models the strategy where the adversary withholds its last block until after
the epoch boundary so that its RANDAO reveal is excluded.  Extends HonestAttack
with the 'M' (missed) realization outcome.
"""
from honest_attack import *
from known_outcome import KnownOutcome
from save_policy import save_xml

from utility_distr import shiftDistr, computeMaxDistribution, compute_Psi_multi
from logger import is_debug, log, set_debug

set_debug(2)
debug = True

class SelfishMixingAttack(HonestAttack):
    
    def __init__(self, parent):  
        self.astring = 'A'
        self.pos=1+getattr(parent, 'pos', 0)
        self.parent = parent
        self.id=len(self.node_list)
        self.node_list.append(self)
        self.realizations = ['O', 'M']
        
    def is_honest(self):
        return False

    
    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "type": "sm",
            "id": self.id,
            "parent": self.parent.attack_string_obj.attack_string if self.parent else None,
        }

    @classmethod
    def from_dict(cls,data):
        parent = None
        if "parent" in data:
             parent = HonestAttack.node_list[data["parent"]]
        # legacy fallback
        elif "parents" in data:
            for pid in data["parents"]:
                if len(HonestAttack.node_list) > pid:
                    parent = HonestAttack.node_list[pid]
                    break
             
        obj = SelfishMixingAttack(parent)
        obj.id = data["id"]
        return obj

if __name__ == "__main__":
    import argparse

    #from saveresults import save_xml

    parser = argparse.ArgumentParser(
        description="Demo of selfish mixing attack strings"
    )
    parser.add_argument(
        "-log",
        help="Show log messages on the command line [1-few,...,5-all].",
        type=int,
    )
    parser.add_argument("-alpha", help="Simulate only alpha", type=int, default=20)
    args = parser.parse_args()
    if not args.log:
        set_debug(2)
    else:
        set_debug(args.log)
    res_log = {"as": [], "alpha": args.alpha * 0.01}
    nodes = []
    def testTree(node):
        node.heuristic_utility()
        nodes.append(node)
        distr = []
        for i, d in enumerate(node.exp_distr):
            distr.append({"id": i, "prob": d})
        res_log["as"].append(
            {
                "attack_string": node.get_attack_string(),
                "disrt": distr,
                "utility": node.utility,
            }
        )
    testTree(HonestAttack(args.alpha * 0.01,None))
    testTree(SelfishMixingAttack(nodes[0]))
    testTree(SelfishMixingAttack(nodes[1]))
    save_xml(res_log, "results")
    import os
    os.system("python  xml_process.py results.xml -y prob -x id -legend attack_string -fig")
