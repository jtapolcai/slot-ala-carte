"""
KnownOutcome: stores the resolved action plan and forking threshold for a
single realization head.

Each KnownOutcome records the realization string (up to 32 uppercase chars),
the continuous forking threshold (t_rho), and an optional value_r for
additional reward accounting.
"""
from logger import is_debug, log, set_debug

class KnownOutcome:
    __slots__ = [
        'realization_str', 'treashold', 'next_real', 'value_r'
    ]
    def __init__(self, realization, next_real=None):
        # Build the full realization string (max 32 chars, uppercase)
        if next_real is not None:
            self.realization_str = (str(realization) + str(next_real)).upper()
        else:
            self.realization_str = str(realization).upper()
        self.next_real=next_real
        #self.realization = realization
        self.treashold = 0.0
        self.value_r = None

    def get_action_plan(self):
        return self.realization_str
    
    def get_treashold(self):
        return self.treashold

    def to_dict(self):
        """Convert the object to a dictionary for JSON serialization."""
        return {
            "realization": self.__repr__(),
            "treashold": self.treashold
        }

    def get_realisation(self, untill=None):
        if untill is None:
            return self.realization_str
        else:
            if untill <= len(self.realization_str):
                return self.realization_str[:untill]
            else:
                log(2, f"Warning! asked for untill {untill} but realization is only {len(self.realization_str)} long.")
                return self.realization_str

    def count_sacrificed(self, untill=None):
        realization_str=self.get_realisation(untill)
        return realization_str.count("M") + realization_str.count("S") + realization_str.count("D")


    def count_dropped(self, untill=None):
        realization_str=self.get_realisation(untill)
        self.value_r=realization_str.count("M") + realization_str.count("P")
        return self.value_r
    # + realization_string.count("O")

    def count_forked(self, till=None):
        realization_string = self.get_realisation(till)
        return realization_string.count("F")
    
    def pos(self):
        return len(self.realization_str)

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["realization"])
        if "treashold" in data:
            obj.treashold = float(data["treashold"])
        return obj

    def __repr__(self) -> str:
        return self.realization_str
