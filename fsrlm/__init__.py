"""
fsRLM - Filesystem-based Recursive Language Model

An RLM-style abstraction that uses Claude Agent SDK as a root agent,
with recursive Claude API calls happening inside Python scripts.
"""

from .rlm import RLM, RLMConfig, RLMResult, invoke, Message, Messages
from .workspace import Workspace, WorkspaceConfig
from .runner import AgentRunner, RunnerConfig

__version__ = "0.1.0"
__all__ = [
    # Main interface
    "RLM",
    "RLMConfig",
    "RLMResult",
    "invoke",
    # Types
    "Message",
    "Messages",
    # Lower-level components
    "Workspace",
    "WorkspaceConfig",
    "AgentRunner",
    "RunnerConfig",
]
