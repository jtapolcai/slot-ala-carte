"""
HonestAttack: base class for per-slot utility computation.

Models the honest (non-forking) publishing strategy and provides the
binomial slot-count distributions shared by all attack types.
ForkingAttack and SelfishMixingAttack both subclass HonestAttack.

Class-level state (set via HonestAttack.set_globals):
    alpha          -- adversary fraction
    basic_distr    -- Binomial(32, alpha) PMF
    basic_cdf      -- corresponding CDF
    limit_sacrifice -- optional sacrifice cap for search-space pruning
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attack_string import AttackString
    
from math import ceil, comb

from utility_distr import compute_Psi_multi, expectedValue, cdf_distr

from logger import is_debug, log, set_debug
from known_outcome import KnownOutcome


set_debug(2)
debug = True


def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))

class HonestAttack:
    # global variables for memory efficiency
    alpha: float = 0
    basic_distr = []
    basic_cdf = []
    # the graph nodes are stored in a list because the order matters
    node_list=[]
    limit_sacrifice=None

    _attack_string_obj: AttackString | None

    @classmethod
    def set_alpha(cls, alpha):
        # set global variables
        cls.alpha = alpha
        cls.basic_distr = []
        for i in range(33):
            cls.basic_distr.append(_binom_pmf(i, 32, alpha))
        cls.basic_cdf = cdf_distr(cls.basic_distr)
        # max_binoms_cache in AttackString is keyed only by j (number of epochs),
        # but the underlying distributions depend on basic_distr (which depends on alpha).
        # Clear it so that a fresh alpha produces correct cached values.
        from attack_string import AttackString
        AttackString.max_binoms_cache.clear()
        AttackString.max_cdf_binoms_cache.clear()

    @classmethod
    def set_limit_sacrifice(cls, limit):
        cls.limit_sacrifice = limit
        if limit is not None:
            log(1,f"Setting minimum sacrifice limit for attack strings to {limit} which limits the search space by {100*cls.get_cdf_binom(ceil(cls.alpha*32)-limit):.4f}%")

    def __init__(self, alpha, attack_string_obj: AttackString | None = None):
        self.astring = '.'
        self.pos=0
        self.parent: HonestAttack | None = None
        self.set_alpha(alpha)
        self.realizations = ['']

        self.id=len(self.node_list)
        self.node_list.append(self)
        self._attack_string_obj = None
        if attack_string_obj is not None:
            self.attack_string_obj = attack_string_obj

    @property
    def attack_string_obj(self) -> AttackString:
        if self._attack_string_obj is None:
            raise RuntimeError("attack_string_obj is not set on HonestAttack")
        return self._attack_string_obj

    @attack_string_obj.setter
    def attack_string_obj(self, value: AttackString) -> None:
        self._attack_string_obj = value

    def is_honest(self):
        return True

    @classmethod
    def get_binom(cls, i):
        if i<0 or i>32:
            return 0
        return cls.basic_distr[i]

    @classmethod
    def get_cdf_binom(cls, i):
        if i<0:
            return 0.0
        if i>32:
            return 1.0
        return cls.basic_cdf[i]

    def compute_realizations(self):
        self.realizations = []
        
    def get_realizations(self) -> list["KnownOutcome"]:
        if self.parent is not None:
            ret = []
            for S in self.parent.get_realizations():
                for s in self.realizations:
                    # Combine local realization (s) with parent realization (S)
                    # s is the 'head' (current), S is the 'tail' (parent)
                    ret.append(KnownOutcome(s, next_real=S))
            return ret
        else:
            return [KnownOutcome(s) for s in self.realizations]

    def get_attack_string(self):
        if self.parent is not None:
             return self.astring + self.parent.get_attack_string()
        return self.astring

    def get_head(self):
        if self.parent:
             return self.parent.get_head()
        parts = self.astring.split(".")
        if len(parts) > 1:
            return parts[1]
        return ""

    def is_over_epoch_boundary(self):
        if self.astring[-1] == ".":
            return False
        return "." in self.get_attack_string()

    def last_slot(self):
        if self.astring == ".":
            return None
        if self.astring[-1] == ".":
            return self.astring[-2]
        return self.astring[-1]


    def __repr__(self):
        """String representation of the TreeNode."""
        return f"{self.astring}"

    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "type": "honest",
            "id": self.id,
            "astring": self.astring,
            "parent": self.parent.attack_string_obj.attack_string if self.parent else  "",
            "alpha": self.alpha
        }

    @classmethod
    def from_dict(cls, data):
        obj = HonestAttack(data["alpha"])
        obj.id = data["id"]
        obj.astring = data["astring"]
        if "parent" in data and data["parent"] is not None:
             pid = data["parent"]
             if pid < len(HonestAttack.node_list):
                obj.parent = HonestAttack.node_list[pid]
        return obj
