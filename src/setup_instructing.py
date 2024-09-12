from typing import List

import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

import util
from cfg import Config
from directory_parsing import DirectoryTree, FileNode
from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class SetupInstructionTool(BaseModel):
    setup_instructions: str = Field(
        description=(
            "'Installation and Environment Setup' section of the README file."
        )
    )

    class Config:
        schema_extra = {
            "required": ["setup_instructions"]
        }


class SetupInstructor(ToolCallingAssistant):
    def __init__(
        self,
        system_prompt: str,
        tools: SetupInstructionTool,
        config: Config,
        max_tool_call_retries: int = 5
    ) -> None:
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state: dict) -> dict:
        repository = dill.loads(state["repository"])
        setup_file_nodes = _get_setup_file_nodes(repository.directory_tree)
        if not setup_file_nodes:
            logger.warning(
                "Failed to locate any files related to installation or environment "
                "setup. The README file is unlikely to contain relevant instructions."
            )
            return {"setup_instructions": None}

        message = _compose_setup_instructor_message_from_nodes(setup_file_nodes)
        try:
            response = self.invoke(message)
        except ToolCallingException as e:
            logger.error(
                "Failed to write instructions for installation and environment setup"
                f"{repr(e)}. The README file is unlikely to contain the relevant "
                "instructions."
            )
            return {"setup_instructions": None}

        tool_output = response.tool_calls[0]["args"]
        setup_instructions = tool_output["setup_instructions"]
        logger.debug("Wrote instructions for installation and environment setup.")
        return {"setup_instructions": setup_instructions}


def _get_setup_file_nodes(directory_tree: DirectoryTree) -> List[FileNode]:
    """
    Extracts and returns a list of setup file nodes from a given directory tree.

    This function iterates over all file nodes in the provided directory tree and
    collects those that are identified as setup files. It utilizes the
    `directory_tree_file_nodes` function to traverse the directory tree and filter nodes
    based on the `is_setup_file` attribute.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree from which to extract setup file nodes.

    Returns
    -------
    List[FileNode]
        A list of file nodes that are setup files within the directory tree.
    """
    setup_file_nodes = []
    for node in util.directory_tree_file_nodes(directory_tree):
        if node.is_setup_file:
            setup_file_nodes.append(node)
    return setup_file_nodes


def _compose_setup_instructor_message_from_nodes(
    setup_file_nodes: List[FileNode]
) -> str:
    """
    Composes a message for the setup instructor by concatenating the contents of
    multiple file nodes.

    Parameters
    ----------
    setup_file_nodes : List[FileNode]
        A list of FileNode objects, each containing a file path and its content.

    Returns
    -------
    str
        A formatted string message that includes the content of each file node, wrapped
    with start and end markers, followed by an instructional note.
    """
    message = ""
    for node in setup_file_nodes:
        message += f"<<START OF FILE: {node.path}>>\n\n"
        message += f"{node.content}\n\n"
        message += "<<END OF FILE>>\n\n\n"

    message += "Please, use the tool provided!"
    return message
