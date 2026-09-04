"""Public method registry used by the single experiment entry point."""

from reproductions.fo import AppleCMS, HadaOracle, MSSOracle, PGROracle, PanOCMS
from reproductions.fo.standard import OLHOracle, UnaryEncodingOracle
from reproductions.hhi import (
    BitstogramCore,
    HadaHeavyCore,
    PrivateExpanderSketchCore,
    TreeHistPaper,
)


HHI_METHODS = {
    "treehist-paper-topk": TreeHistPaper,
    "bitstogram-core": BitstogramCore,
    "hadaheavy-core": HadaHeavyCore,
    "privateexpander-core": PrivateExpanderSketchCore,
}

FO_METHODS = {
    "olh": OLHOracle,
    "oue": lambda: UnaryEncodingOracle(True),
    "sue": lambda: UnaryEncodingOracle(False),
    "cms": lambda: AppleCMS(False),
    "hcms": lambda: AppleCMS(True),
    "hadaoracle-core": HadaOracle,
    "pgr-python-port": PGROracle,
    "mss": MSSOracle,
    "pan-ocms-independent": PanOCMS,
}
