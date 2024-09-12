import textwrap

import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

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
    def __init__(self, system_prompt, tools, config, max_tool_call_retries=5):
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state):
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
    node, call_chain_context_up=None, call_chain_context_down=None
):
    """
    Retrieves the proximate nodes in a call graph by traversing both upward and downward
    from a given node.

    Parameters
    ----------
    node : Node
        The starting node from which to traverse the call graph.
    call_chain_context_up : int, optional
        The maximum depth to traverse upward. If None, traverses all ancestors.
    call_chain_context_down : int, optional
        The maximum depth to traverse downward. If None, traverses all descendants.

    Returns
    -------
    list of Node
        A list of unique nodes that are either ancestors or descendants of the starting
    node, up to the specified depths.
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
    node, call_chain_context_up, call_chain_context_down
):
    """
    Retrieves the definitions of proximate nodes in a call graph, excluding those that
    are part of the class definition of the current node.

    Parameters
    ----------
    node : Node
        The node from which to retrieve proximate node definitions.
    call_chain_context_up : int
        The maximum depth to traverse upward in the call graph.
    call_chain_context_down : int
        The maximum depth to traverse downward in the call graph.

    Returns
    -------
    list of str
        A list of dedented string definitions of proximate nodes, excluding those whose
    signatures are found in the class definition of the current node.
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


def _compose_docstring_writer_message_from_node(node, user_context, config):
    """
    Composes a detailed message for generating a new docstring and summary for a given
    function node.

    This function constructs a message that includes the current function's definition,
    any user-provided context, and definitions of proximate functions in the call chain.
    It also includes the class definition if the function is a method. The message is
    intended to guide the creation of a new docstring and summary.

    Parameters
    ----------
    node : Node
        The node representing the function for which the docstring and summary are to be
    generated.
    user_context : str or None
        Optional high-level context provided by the user about the repository.
    config : Config
        Configuration object containing parameters for call chain context.

    Returns
    -------
    str
        A composed message string that includes the function definition, user context,
    proximate function definitions, and class definition if applicable."""

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


def _traverse_by_depth_reversed(call_graph):
    """
    Traverse the nodes of a call graph in reverse order of their depth.

    This function iterates over the nodes of a call graph, yielding nodes grouped by
    their depth in descending order. It first checks if all nodes have a computed depth
    value, raising a RuntimeError if any node lacks this information. It then organizes
    nodes into a dictionary keyed by their depth and iterates from the maximum depth to
    zero, yielding nodes at each depth level.

    Parameters
    ----------
    call_graph : CallGraph
        The call graph containing nodes to be traversed. Each node must have a 'depth'
    attribute.

    Yields
    ------
    list of Node
        A list of nodes at the current depth level, starting from the maximum depth and
    proceeding to zero.

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
