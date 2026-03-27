from .generator import *
from .attention import *

try:
    from .visualization import *
except ModuleNotFoundError:
    ISAVisualization = None

InterSentenceAttention = ISA
InterSentenceAttentionVisualization = ISAVisualization
