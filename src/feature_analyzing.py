from typing import List, Union

import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

import util
from cfg import Config
from directory_parsing import DirectoryTree, FileNode
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
    def __init__(
        self,
        system_prompt: str,
        tools: FeatureAnalyzingTool,
        config: Config,
        max_tool_call_retries: int = 5
    ) -> None:
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state: dict) -> dict:
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


def _get_py_file_nodes(directory_tree: DirectoryTree) -> List[FileNode]:
    """
    Extracts and returns a list of Python file nodes from a given directory tree.

    This function iterates over all file nodes in the provided directory tree and
    filters out those that represent Python files. It utilizes the
    `directory_tree_file_nodes` function to access all file nodes and checks each node's
    `is_py_file` attribute to determine if it is a Python file.

    Parameters
    ----------
    directory_tree : DirectoryTree
        An instance of `DirectoryTree` from which Python file nodes are to be extracted.

    Returns
    -------
    List[FileNode]
        A list of `FileNode` objects that represent Python files within the directory
    tree.
    """
    py_file_nodes = []
    for node in util.directory_tree_file_nodes(directory_tree):
        if node.is_py_file:
            py_file_nodes.append(node)
    return py_file_nodes


def _compose_feature_analyzer_message_from_nodes(
    py_file_nodes: List[FileNode], user_context: Union[str, None]
) -> str:
    """
    Composes a feature analyzer message from a list of file nodes and optional user
    context.

    This function constructs a message string that includes summaries of Python files
    represented by `FileNode` objects. If a user context is provided, it is included at
    the beginning of the message. Each file node's path and summary are appended to the
    message, followed by a prompt to use the provided tool.

    Parameters
    ----------
    py_file_nodes : List[FileNode]
        A list of `FileNode` objects, each representing a Python file with a path and
    summary.
    user_context : Union[str, None]
        An optional string providing high-level context about the repository. If None,
    no context is added.

    Returns
    -------
    str
        A composed message string containing the user context (if provided) and
    summaries of the specified file nodes.
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
