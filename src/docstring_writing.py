import textwrap
from typing import Iterator, List, Optional, Union

import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

from call_graph_parsing import CallGraph, CallGraphNode
from cfg import Config
from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class DocstringWritingTool(BaseModel):
    docstring: str = Field(
        description=(
            "New docstring for the function without column offset or maximumwidth"
        )
    )
    summary: str = Field(
        description="High-level summary of the function's purpose and operation."
    )

    class Config:
        schema_extra = {
            "required": ["docstring", "summary"]
        }


class DocstringWriter(ToolCallingAssistant):
    def __init__(
        self,
        system_prompt: str,
        tools: DocstringWritingTool,
        config: Config,
        max_tool_call_retries: int = 5,
    ) -> None:
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state: dict) -> dict:
        user_context = state["user_context"]
        repository = dill.loads(state["repository"])
        for nodes_at_depth in _traverse_by_depth_reversed(repository.call_graph):
            # process one node at a time
            for node in nodes_at_depth:
                message = _compose_docstring_writer_message_from_node(
                    node, user_context, self.config
                )
                try:
                    response = self.invoke(message)
                except ToolCallingException as e:
                    logger.error(
                        f"Could not write docstring to function '{node.name}': "
                        f"{repr(e)}"
                    )
                    continue
                tool_output = response.tool_calls[0]["args"]
                node.add_new_docstring(tool_output["docstring"])
                node.insert_new_docstring(
                    self.config.max_width, self.config.indent_size
                )
                node.add_function_summary(tool_output["summary"])
                logger.debug(f"Wrote docstring to function '{node.name}'")
        return {"repository": dill.dumps(repository)}


def _get_proximate_nodes(
    node: CallGraphNode,
    call_chain_context_up: Optional[int] = None,
    call_chain_context_down: Optional[int] = None
) -> List[CallGraphNode]:
    """
    Retrieves nodes that are proximate to a given node in a call graph, including both
    parent and child nodes up to specified depths.

    This function collects nodes that are either ancestors or descendants of the
    specified `node` within a call graph, based on the provided depth limits for upward
    and downward traversal. It combines these nodes into a single list, ensuring no
    duplicates.

    Parameters
    ----------
    node : CallGraphNode
        The node from which to find proximate nodes in the call graph.
    call_chain_context_up : Optional[int], optional
        The maximum depth for traversing upwards to find parent nodes. If None, all
    ancestors are included.
    call_chain_context_down : Optional[int], optional
        The maximum depth for traversing downwards to find child nodes. If None, all
    descendants are included.

    Returns
    -------
    List[CallGraphNode]
        A list of unique `CallGraphNode` objects that are either parents or children of
    the given node, up to the specified depths.
    """
    parent_nodes = [
        node for depth, node in node.traverse_up(
            max_depth=call_chain_context_up, return_start_node=False
        )
    ]
    child_nodes = [
        node for depth, node in node.traverse_down(
            max_depth=call_chain_context_down, return_start_node=False
        )
    ]
    return list(set(parent_nodes) | set(child_nodes))


def _get_proximate_node_definitions(
    node: CallGraphNode,
    call_chain_context_up: Optional[int] = None,
    call_chain_context_down: Optional[int] = None,
) -> List[str]:
    """
    Retrieves the definitions of nodes that are proximate to a given node in a call
    graph, excluding those whose signatures are found in the class definition.

    This function first identifies nodes that are proximate to the specified `node` by
    calling `_get_proximate_nodes`, which considers both upward and downward traversal
    in the call graph. It then extracts and dedents the definitions of these nodes. If
    the `node` is associated with a class, it filters out any node definitions whose
    signatures are present in the class definition to avoid redundancy.

    Parameters
    ----------
    node : CallGraphNode
        The node for which proximate node definitions are to be retrieved.
    call_chain_context_up : Optional[int], optional
        The maximum depth for traversing upwards to find parent nodes. If None, all
    ancestors are included.
    call_chain_context_down : Optional[int], optional
        The maximum depth for traversing downwards to find child nodes. If None, all
    descendants are included.

    Returns
    -------
    List[str]
        A list of dedented string definitions of proximate nodes, excluding those whose
    signatures are found in the class definition.
    """
    proximate_nodes = _get_proximate_nodes(
        node, call_chain_context_up, call_chain_context_down
    )
    proximate_node_definitions = [
        textwrap.dedent(n.definition) for n in proximate_nodes
    ]

    # remove functions whose signature is found in the class definition
    if node.class_node:
        proximate_node_definitions = [
            d for d in proximate_node_definitions
            if d.splitlines()[0].strip() not in node.class_node.definition
        ]

    return proximate_node_definitions


def _compose_docstring_writer_message_from_node(
    node: CallGraphNode, user_context: Union[str, None], config: Config
) -> str:
    """
    Composes a detailed message for generating a new docstring and summary for a
    function node in a call graph.

    This function constructs a message that includes the current function's definition,
    any user-provided context, and definitions of proximate functions. It also includes
    the class definition if the function is a method. The message is formatted to guide
    the creation of a new docstring and summary for the function.

    Parameters
    ----------
    node : CallGraphNode
        The node representing the function for which the message is being composed.
    user_context : Union[str, None]
        Optional user-provided context about the repository, which is included in the
    message if available.
    config : Config
        Configuration object containing settings for call chain context, such as the
    depth of context to include.

    Returns
    -------
    str
        A formatted message string that includes the function definition, user context,
    proximate function definitions, and class definition if applicable, intended to
    assist in writing a new docstring and summary."""
    proximate_node_definitions = _get_proximate_node_definitions(
        node, config.call_chain_context_up, config.call_chain_context_down
    )

    message = ""

    if user_context is not None:
        message += "User-provided high-level context of repository:\n\n"
        message += f"'{user_context}'\n\n\n"

    message += "Current Function Definition (with old docstring if available):\n\n"
    message += f"{textwrap.dedent(node.definition)}\n\n\n"

    if proximate_node_definitions:
        message += "Proximate Function Definitions:\n\n"
        for pf_definition in proximate_node_definitions:
            message += f"{pf_definition}\n\n\n"

    if node.class_node:
        message += "Class Definition of Current Method:\n\n"
        message += f"{node.class_node.definition}\n\n\n"

    message += (
        "Please write a new, detailed docstring for the current function, considering "
        "the provided context and related function definitions. "
        "Additionally, provide an intuitive summary of the function's purpose "
        "and how it achieves that purpose. \n\n"
        "Remember, do NOT split lines in the docstring. Ensure that text, parameters, "
        "and return values are kept on single lines without forced line breaks."
        "Lastly, remember to use the tool provided!"
    )

    return message


def _traverse_by_depth_reversed(call_graph: CallGraph) -> Iterator[List[CallGraph]]:
    """
    Traverse the call graph by depth in reverse order, yielding nodes at each depth
    level.

    Parameters
    ----------
    call_graph : CallGraph
        The call graph to traverse, which must have depth values computed for all nodes.

    Yields
    ------
    Iterator[List[CallGraph]]
        An iterator over lists of nodes, where each list contains nodes at the same
    depth level, starting from the maximum depth to zero.

    Raises
    ------
    RuntimeError
        If any node in the call graph does not have a depth value computed.
    """
    if any(node.depth is None for node in call_graph.nodes.values()):
        raise RuntimeError(
            "Attempted to traverse CallGraph by depths but all nodes do not have depth "
            "value computed."
        )

    depth_dict = {}
    for node in call_graph.nodes.values():
        if node.depth not in depth_dict:
            depth_dict[node.depth] = []
        depth_dict[node.depth].append(node)

    max_depth = max(depth_dict.keys())
    for depth in range(max_depth, -1, -1):
        yield depth_dict[depth]
