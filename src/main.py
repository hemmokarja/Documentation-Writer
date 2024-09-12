import argparse

from app import DocumentationWriter
from call_graph_parsing import CallGraph
from cfg import Config
from directory_parsing import DirectoryTree
from util import Repository


def parse_args():
    """
    Parses command-line arguments for configuring the documentation generation process.

    This function sets up an argument parser to handle various command-line options that
    control the behavior of the documentation writer. It includes options for specifying
    the directory to process, user context, return mode, model settings, output
    formatting, and integration with LangSmith and LangChain.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Path to the directory to be processed."
    )
    parser.add_argument(
        "--user_context",
        type=str,
        default=None,
        help="Optional user-provided high-level context of the repository."
    )
    parser.add_argument(
        "--return_mode",
        type=str,
        default="create_new",
        choices=["create_new", "modify_existing"],
        help=(
            "Return mode for the results. Must be either 'create_new' (returns a new "
            "directroy of Python files with docstrings and README file) or "
            "'modify_existing' (modifies existing Python files and creates README in "
            "the same directory)"
        )
    )
    parser.add_argument(
        "--new_dir_location",
        type=str,
        default="documentation-writer-result-dir",
        help="Location of the new directory when `return_mode=='create_new'`."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Name of the used OpenAI model."
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=0,
        help="Temperature setting for model inference."
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=88,
        help="Maximum width for output formatting."
    )
    parser.add_argument(
        "--indent_size",
        type=int,
        default=4,
        help="Indent size for output formatting."
    )
    parser.add_argument(
        "--call_chain_context_up",
        type=int,
        default=2,
        help=(
            "Call chain context windown before processed function in docstring "
            "generation."
        )
    )
    parser.add_argument(
        "--call_chain_context_down",
        type=int,
        default=2,
        help=(
            "Call chain context windown after processed function in docstring "
            "generation."
        )
    )
    parser.add_argument(
        "--use_langsmith",
        type=bool,
        default=True,
        help="Flag to use LangSmith."
    )
    parser.add_argument(
        "--langchain_project",
        type=str,
        default="documentation-writer",
        help="LangChain project name."
    )
    parser.add_argument(
        "--langchain_tracing_v2",
        type=str,
        default="true",
        help="Enable LangChain tracing v2."
    )
    parser.add_argument(
        "--langchain_endpoint",
        type=str,
        default="https://api.smith.langchain.com",
        help="LangSmith endpoint."
    )
    parser.add_argument(
        "--langchain_user_agent",
        type=str,
        default="myagent",
        help="User agent for LangChain API calls."
    )

    return parser.parse_args()


def run():
    """
    Executes the main process of the documentation generation application.

    This function orchestrates the entire workflow of the documentation generation
    process. It begins by parsing command-line arguments to configure the application
    settings. It then initializes a `Config` object with these settings and constructs a
    `DirectoryTree` to represent the file structure of the specified directory. The
    directory tree is parsed to identify the structure and contents of the directory. A
    `CallGraph` is then created and populated by parsing the directory tree to map
    function calls and relationships. A `Repository` object is instantiated to
    encapsulate the directory tree and call graph. Finally, a `DocumentationWriter`
    application is initialized and executed to process the repository and generate
    documentation based on the configuration.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
    """
    args = parse_args()
    config = Config(
        user_context=args.user_context,
        directory=args.directory,
        return_mode=args.return_mode,
        new_dir_location=args.new_dir_location,
        model_name=args.model_name,
        temperature=args.temperature,
        max_width=args.max_width,
        indent_size=args.indent_size,
        call_chain_context_up=args.call_chain_context_up,
        call_chain_context_down=args.call_chain_context_down,
        use_langsmith=args.use_langsmith,
        langchain_project=args.langchain_project,
        langchain_tracing_v2=args.langchain_tracing_v2,
        langchain_endpoint=args.langchain_endpoint,
        langchain_user_agent=args.langchain_user_agent
    )
    directory_tree = DirectoryTree()
    directory_tree.parse_tree(config.directory)
    call_graph = CallGraph()
    call_graph.parse_graph(directory_tree)
    repository = Repository(directory_tree, call_graph)
    app = DocumentationWriter(config)
    app.run_app(repository, config)


if __name__ == "__main__":
    run()
