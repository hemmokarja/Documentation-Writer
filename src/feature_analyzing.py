import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

import util
from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class FeatureAnalyzingTool(BaseModel):
    feature_summary: str = Field(
        description="Summary of the repository's core functionalities and features."
    )

    class Config:
        schema_extra = {
            "required": ["feature_summary"]
        }


class FeatureAnalyzer(ToolCallingAssistant):
    def __init__(self, system_prompt, tools, config, max_tool_call_retries=5):
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state):
        user_context = state["user_context"]
        repository = dill.loads(state["repository"])
        py_file_nodes = _get_py_file_nodes(repository.directory_tree)
        message = _compose_feature_analyzer_message_from_nodes(
            py_file_nodes, user_context
        )
        try:
            response = self.invoke(message)
        except ToolCallingException as e:
            logger.error(
                f"Failed to write summary of repository's key features: {repr(e)} "
                "Relevant information is likely to be omitted from the README file."
            )
        tool_output = response.tool_calls[0]["args"]
        feature_summary = tool_output["feature_summary"]
        logger.debug("Wrote summary of repository's key features")
        return {"feature_summary": feature_summary}


def _get_py_file_nodes(directory_tree):
    """
    Extracts Python file nodes from a directory tree.

    This function iterates over all file nodes in the given directory tree and collects
    those that are Python files. It utilizes the `directory_tree_file_nodes` function to
    access each file node and checks if the node represents a Python file by evaluating
    the `is_py_file` attribute.

    Parameters
    ----------
    directory_tree : object
        An object representing the directory tree, which must have a `walk` method that
    yields directory paths, subdirectories, and file nodes.

    Returns
    -------
    list of FileNode
        A list of file nodes that represent Python files within the directory tree.
    """
    py_file_nodes = []
    for node in util.directory_tree_file_nodes(directory_tree):
        if node.is_py_file:
            py_file_nodes.append(node)
    return py_file_nodes


def _compose_feature_analyzer_message_from_nodes(py_file_nodes, user_context):
    """
    Constructs a feature analyzer message from a list of Python file nodes and optional
    user context.

    This function generates a message that includes a user-provided context about the
    repository, if available, followed by summaries of each Python file node. The
    message concludes with a prompt to use a specific tool.

    Parameters
    ----------
    py_file_nodes : list
        A list of nodes, where each node represents a Python file and contains a 'path'
    and 'summary'.
    user_context : str or None
        An optional string providing high-level context about the repository.

    Returns
    -------
    str
        A composed message string that includes the user context and file summaries.
    """
    message = ""

    if user_context is not None:
        message += "User-provided high-level context of repository:\n\n"
        message += f"'{user_context}'\n\n\n"

    for node in py_file_nodes:
        message += f"<<SUMMARY FOR FILE: {node.path}>>\n\n"
        message += f"{node.summary}\n\n\n"

    message += "Please, use the tool provided!"
    return message
