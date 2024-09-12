import os
from typing import Iterator, List

import structlog
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from call_graph_parsing import CallGraph
from cfg import Config
from directory_parsing import DirectoryTree, FileNode

logger = structlog.get_logger(__name__)


class Repository:
    # wrapper for passing directory tree and call graph between langgraph nodes in
    # serialized format while preserving the references between shared FileNodes
    def __init__(self, directory_tree: DirectoryTree, call_graph: CallGraph) -> None:
        self.directory_tree = directory_tree
        self.call_graph = call_graph


class ToolCallingException(BaseException):
    pass


class ToolCallingAssistant:
    def __init__(
        self,
        system_prompt_path: str,
        tools: BaseModel,
        config: Config,
        max_tool_call_retries: int = 5
    ) -> None:
        if not isinstance(tools, list):
            tools = [tools]

        self.system_prompt = _read_system_prompt(system_prompt_path)
        self.tools = tools
        self.config = config
        self.max_tool_call_retries = max_tool_call_retries
        self.runnable = _init_tool_assistant_runnable(
            self.system_prompt, config.model_name, config.temperature, tools
        )
        self.subclass_name = type(self).__name__

    def invoke(self, message_content: str) -> AIMessage:
        """
        Attempts to invoke a tool using the provided message content, retrying if
        necessary.

        This method sends a message to a tool and expects a single tool call in
        response. If the tool call fails or results in multiple calls, it retries up to
        a specified maximum number of attempts. If the tool call is successful, it
        returns the response. If all attempts fail, it raises a ToolCallingException.

        Parameters
        ----------
        message_content : str
            The content of the message to be sent to the tool.

        Returns
        -------
        AIMessage
            The response from the tool if a single tool call is successful.

        Raises
        ------
        ToolCallingException
            If the tool fails to be called successfully after the maximum number of
        retries.
        """
        messages = [HumanMessage(content=message_content)]

        for i in range(self.max_tool_call_retries):
            response = self.runnable.invoke({"messages": messages})
            num_tool_calls = len(response.tool_calls)
            if num_tool_calls == 1:
                return response
            if num_tool_calls > 1:
                logger.warning(
                    f"{self.subclass_name} called tool {num_tool_calls} times although "
                    "one call was expected."
                )
            if num_tool_calls == 0:
                logger.warning(
                    f"{self.subclass_name} failed calling tool on attempt {i+1}/"
                    f"{self.max_tool_call_retries}. Trying again..."
                )
                messages += [HumanMessage(content="Please, use the tool provided.")]
                continue
        raise ToolCallingException(
            f"{self.subclass_name} failed to call tool after "
            f"{self.max_tool_call_retries} attempts."
        )


def _init_tool_assistant_runnable(
    system_prompt: str, model_name: str, temperature: float, tools: List[BaseModel]
) -> Runnable:
    """
    Initializes a tool-assisted runnable for a language model with a specified system
    prompt and configuration.

    This function creates a chat prompt template using the provided system prompt and
    initializes a language model with the specified model name and temperature. It then
    binds the given tools to the language model, ensuring that the tools are used in a
    strict and non-parallel manner. The resulting runnable is a combination of the chat
    prompt and the tool-bound language model.

    Parameters
    ----------
    system_prompt : str
        The system prompt to be used in the chat prompt template.
    model_name : str
        The name of the language model to be initialized.
    temperature : float
        The temperature setting for the language model, affecting the randomness of its
    outputs.
    tools : List[BaseModel]
        A list of tools to be bound to the language model.

    Returns
    -------
    Runnable
        A runnable object that combines the chat prompt template and the tool-bound
    language model.
    """
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("messages")
        ]
    )
    llm = ChatOpenAI(
        model_name=model_name, temperature=temperature
    )
    llm_with_tools = llm.bind_tools(
        tools=tools, strict=True, parallel_tool_calls=False
    )
    return assistant_prompt | llm_with_tools


def _read_system_prompt(filename: str, prompt_dir: str = "prompts") -> str:
    """
    Reads the content of a system prompt file from a specified directory.

    Parameters
    ----------
    filename : str
        The name of the file containing the system prompt.
    prompt_dir : str, optional
        The directory where the prompt file is located, relative to the script
    directory. Defaults to 'prompts'.

    Returns
    -------
    str
        The content of the system prompt file as a string.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist in the given directory.
    IOError
        If there is an error reading the file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, prompt_dir, filename)
    with open(path, "r") as f:
        prompt = f.read()
    return prompt


def set_langchain_env(config: Config) -> None:
    """
    Sets up the environment variables required for LangSmith by validating and assigning
    values from the provided configuration.

    Parameters
    ----------
    config : Config
        A configuration object containing the necessary settings for LangSmith,
    including `langchain_project`, `langchain_tracing_v2`, `langchain_endpoint`, and
    `langchain_user_agent`.

    Raises
    ------
    ValueError
        If any of the required environment variables (`LANGCHAIN_API_KEY`,
    `langchain_project`, `langchain_tracing_v2`, `langchain_endpoint`,
    `langchain_user_agent`) are not set in the configuration or environment.

    Notes
    -----
    This function ensures that all necessary environment variables for LangSmith are
    set, and logs a confirmation message once the setup is complete."""
    if os.environ.get("LANGCHAIN_API_KEY") is None:
        raise ValueError(
            "Set `LANGCHAIN_API_KEY` as environment variable or set "
            "`Config.use_langsmith` to False"
        )
    if config.langchain_project is None:
        raise ValueError("`Config.langchain_project` cannot be None")
    if config.langchain_tracing_v2 is None:
        raise ValueError("`Config.langchain_tracing_v2` cannot be None")
    if config.langchain_endpoint is None:
        raise ValueError("`Config.langchain_endpoint` cannot be None")
    if config.langchain_user_agent is None:
        raise ValueError("`Config.langchain_user_agent` cannot be None")
    os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
    os.environ["LANGCHAIN_TRACING_V2"] = config.langchain_tracing_v2
    os.environ["LANGCHAIN_ENDPOINT"] = config.langchain_endpoint
    os.environ["USER_AGENT"] = config.langchain_user_agent
    logger.info("LangSmith environment set up!")


def check_openai_env() -> None:
    """
    Checks for the presence of the 'OPENAI_API_KEY' environment variable.

    This function verifies whether the 'OPENAI_API_KEY' is set in the environment
    variables. If the key is not found, it raises a ValueError, indicating that the
    environment variable must be set for the application to function correctly.

    Raises
    ------
    ValueError
        If the 'OPENAI_API_KEY' environment variable is not set."""
    if os.environ.get("OPENAI_API_KEY") is None:
        raise ValueError("Set `OPENAI_API_KEY` as environment variable")


def directory_tree_file_nodes(directory_tree: DirectoryTree) -> Iterator[FileNode]:
    """
    Yields file nodes from a given directory tree.

    This function traverses a directory tree and yields each file node encountered. It
    utilizes the `walk` method of the `DirectoryTree` class to perform a breadth-first
    traversal, iterating over each directory and its files.

    Parameters
    ----------
    directory_tree : DirectoryTree
        An instance of `DirectoryTree` that provides a method to traverse and yield file
    nodes.

    Yields
    ------
    FileNode
        Each file node found in the directory tree during the traversal.
    """
    for _, _, file_nodes in directory_tree.walk():
        for node in file_nodes:
            yield node
