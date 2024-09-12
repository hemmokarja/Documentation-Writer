import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

import util
from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class FileAnalyzingTool(BaseModel):
    summary: str = Field(
        description="Summary of the file's core functionality and purpose."
    )
    is_setup_file: bool = Field(
        description=(
            "True/False whether whether file is relevant for 'Installation' or "
            "'Environment Setup' sections of README.md"
        )
    )
    is_entrypoint_file: bool = Field(
        description=(
            "True/False whether file is likely to contain an entrypoint to the "
            "application"
        )
    )

    class Config:
        schema_extra = {
            "required": ["summary", "is_setup_file", "is_entrypoint_file"]
        }


class FileAnalyzer(ToolCallingAssistant):
    def __init__(self, system_prompt, tools, config, max_tool_call_retries=5):
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state):
        user_context = state["user_context"]
        repository = dill.loads(state["repository"])
        for node in util.directory_tree_file_nodes(repository.directory_tree):
            if not node.is_text_file:
                message = _compose_file_analyzer_message_from_node(node, user_context)
                try:
                    response = self.invoke(message)
                except ToolCallingException as e:
                    logger.error(
                        f"Failed to analyze file '{node.path}': {repr(e)}. Skipping "
                        "file. Note, that this may degrade the quality of the "
                        "README.md file."
                    )
                    continue
                tool_output = response.tool_calls[0]["args"]
                is_setup_file = tool_output["is_setup_file"]
                is_entrypoint_file = tool_output["is_entrypoint_file"]
                node.add_file_summary(tool_output["summary"])
                node.mark_whether_setup_file(is_setup_file)
                node.mark_whether_entrypoint_file(is_entrypoint_file)
                logger.debug(
                    f"Analyzed file '{node.path}' ",
                    is_setup_file=is_setup_file,
                    is_entrypoint_file=is_entrypoint_file,
                )
        return {"repository": dill.dumps(repository)}


def _compose_file_analyzer_message_from_node(node, user_context):
    """
    Constructs a detailed message string from a given node and optional user context.

    This function generates a message that includes user-provided context, if available,
    and details about a file represented by the node, such as its path and content. The
    message is formatted to include clear delimiters for the start and end of the file
    content.

    Parameters
    ----------
    node : object
        An object representing a file, which must have 'path' and 'content' attributes.
    user_context : str or None
        Optional high-level context provided by the user about the repository.

    Returns
    -------
    str
        A formatted message string containing the user context, file path, and file
    content.
    """
    message = ""

    if user_context is not None:
        message += "User-provided high-level context of repository:\n\n"
        message += f"'{user_context}'\n\n\n"

    message += f"The filepath of analyzed file: '{node.path}'\n\n\n"

    message += "The contents of the analyzed file:\n\n"
    message += "<<START OF FILE>>\n\n"
    message += f"{node.content}\n\n"
    message += "<<END OF FILE>>"

    message += "Please, use the tool provided!"
    return message
