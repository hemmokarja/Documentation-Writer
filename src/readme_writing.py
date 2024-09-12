from typing import Union

import dill
import structlog
from langchain_core.pydantic_v1 import BaseModel, Field

import util
from cfg import Config
from directory_parsing import DirectoryTree
from util import ToolCallingAssistant, ToolCallingException

logger = structlog.get_logger(__name__)


class ReadMeWritingTool(BaseModel):
    readme: str = Field(description="The final README file.")

    class Config:
        schema_extra = {
            "required": ["readme"]
        }


class ReadMeWriter(ToolCallingAssistant):
    def __init__(
        self,
        system_prompt: str,
        tools: ReadMeWritingTool,
        config: Config,
        max_tool_call_retries: int = 5
    ) -> None:
        super().__init__(system_prompt, tools, config, max_tool_call_retries)

    def __call__(self, state: dict) -> dict:
        user_context = state["user_context"]
        feature_summary = state["feature_summary"]
        setup_instructions = state["setup_instructions"]
        usage_instructions = state["usage_instructions"]
        repository = dill.loads(state["repository"])
        license_summary = _find_lincense_summary(repository.directory_tree)
        if not any(
            [
                user_context,
                feature_summary,
                setup_instructions,
                usage_instructions,
                license_summary
            ]
        ):
            logger.error(
                "GraphState fields `user_context`, `feature_summary`, "
                "`setup_instructions`, `usage_instructions`, and `license_summary` are "
                "all `None` and thus ReadMeWriter is unable to write the README file."
            )

        message = _compose_readme_writer_message(
            user_context,
            feature_summary,
            setup_instructions,
            usage_instructions,
            license_summary,
        )
        try:
            response = self.invoke(message)
        except ToolCallingException as e:
            logger.error(
                f"Failed to write the final README file for repository: {repr(e)} "
            )

        tool_output = response.tool_calls[0]["args"]
        readme = tool_output["readme"]
        logger.info("Wrote the README file for the repository.")
        return {"readme": readme}


def _find_lincense_summary(directory_tree: DirectoryTree) -> Union[str, None]:
    """
    Searches for and retrieves the summary of a license file from a directory tree.

    This function traverses the file nodes within a given directory tree to identify a
    file node marked as a license. Upon finding such a node, it returns the summary of
    the license file. If no license file is detected, the function returns None.

    Parameters
    ----------
    directory_tree : DirectoryTree
        An instance of `DirectoryTree` that provides a method to traverse and yield file
    nodes.

    Returns
    -------
    Union[str, None]
        The summary of the license file if a license node is found, otherwise None.
    """
    for node in util.directory_tree_file_nodes(directory_tree):
        if node.is_license:
            return node.summary
    return None


def _compose_readme_writer_message(
    user_context: Union[str, None],
    feature_summary: Union[str, None],
    setup_instructions: Union[str, None],
    usage_instructions: Union[str, None],
    license_summary: Union[str, None],
) -> str:
    """
    Composes a README writer message by concatenating various sections of information.

    This function takes optional strings for user context, feature summary, setup
    instructions, usage instructions, and license summary, and constructs a message by
    appending each non-None section with appropriate headers.

    Parameters
    ----------
    user_context : Union[str, None]
        A high-level context of the repository provided by the user.
    feature_summary : Union[str, None]
        A summary of the features of the repository.
    setup_instructions : Union[str, None]
        Instructions on how to set up the repository.
    usage_instructions : Union[str, None]
        Instructions on how to use the repository.
    license_summary : Union[str, None]
        A summary of the repository's license.

    Returns
    -------
    str
        A composed message containing the provided sections with headers, followed by a
    prompt to use the tool provided.
    """
    message = ""

    if user_context is not None:
        message += "User-provided high-level context of repository:\n\n"
        message += f"'{user_context}'\n\n\n"

    if feature_summary is not None:
        message += "<<START OF FEATURE SUMMARY>>\n\n"
        message += f"{feature_summary}\n\n\n"

    if setup_instructions is not None:
        message += "<<START OF SETUP INSTRUCTIONS>>\n\n"
        message += f"{setup_instructions}\n\n\n"

    if usage_instructions is not None:
        message += "<<START OF USAGE INSTRUCTIONS>>\n\n"
        message += f"{usage_instructions}\n\n\n"

    if license_summary is not None:
        message += "<<START OF LICENSE SUMMARY>>\n\n"
        message += f"{usage_instructions}\n\n\n"

    message += "Please use the tool provided!"
    return message
