import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class UsageInstructionTool(BaseModel):
    usage_instructions: str = Field(
        description="'Example Usage' section of the README file"
    )

    class Config:
        schema_extra = {
            "required": ["usage_instructions"]
        }


class UsageInstructor(ToolCallingAssistant):
    def __init__(self, system_prompt, tools, config, max_tool_call_retries=5):
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state):
        repository = dill.loads(state["repository"])
        entrypoint_file_nodes = _get_entrypoint_file_nodes(repository.call_graph)
        if not entrypoint_file_nodes:
            logger.warning(
                "Failed to identify entrypoints to the application. As a result, the "
                "README file is unlikely to contain section for usage instructions."
            )
            return {"usage_instructions": None}

        message = _compose_usage_instructor_message_from_nodes(
            entrypoint_file_nodes, state["setup_instructions"]
        )
        try:
            response = self.invoke(message)
        except ToolCallingException as e:
            logger.error(
                f"Failed to write instructions for example usage: {repr(e)}. The "
                "README file is unlikely to contain the relevant instructions."
            )
            return {"usage_instructions": None}

        tool_output = response.tool_calls[0]["args"]
        usage_instructions = tool_output["usage_instructions"]
        logger.debug("Wrote instructions for example usage.")
        return {"usage_instructions": usage_instructions}


def _get_entrypoint_file_nodes(call_graph):
    """
    Identify and return a set of entry point file nodes from the call graph.

    This function iterates over the entry point nodes in the call graph, as retrieved by
    the `get_entrypoints` method, and collects those nodes whose associated file nodes
    are marked as entry point files.

    Parameters
    ----------
    call_graph : CallGraph
        The call graph from which entry point file nodes are to be identified.

    Returns
    -------
    set of FileNode
        A set of file nodes that are marked as entry point files in the call graph.
    """
    entrypoint_file_nodes = set()
    for function_node in call_graph.get_entrypoints():
        if function_node.file_node.is_entrypoint_file:
            entrypoint_file_nodes.add(function_node.file_node)
    return entrypoint_file_nodes


def _compose_usage_instructor_message_from_nodes(
    entrypoint_file_nodes, setup_instructions
):
    """
    Composes a usage instruction message from a list of file nodes and optional setup
    instructions.

    This function constructs a message string that includes setup instructions, if
    provided, and the content of each file node. Each file node's content is wrapped
    with markers indicating the start and end of the file content. The message concludes
    with a prompt to use the provided tool.

    Parameters
    ----------
    entrypoint_file_nodes : list
        A list of file nodes, where each node contains a 'path' and 'content' attribute
    representing the file's path and its content, respectively.
    setup_instructions : str or None
        Optional setup instructions to be included at the beginning of the message. If
    None, no setup instructions are added.

    Returns
    -------
    str
        A formatted message string containing the setup instructions and file contents,
    ready for display or further processing.
    """
    message = ""
    if setup_instructions is not None:
        message += "<<START OF SETUP INSTRUCTIONS>>\n\n"
        message += f"{setup_instructions}\n\n"
        message += "<<END OF SETUP INSTRUCTIONS>>\n\n"

    for node in entrypoint_file_nodes:
        message += f"<<START OF FILE: {node.path}>>\n\n"
        message += f"{node.content}\n\n"
        message += "<<END OF FILE>>\n\n\n"

    message += "Please, use the tool provided!"
    return message
