from typing import Optional, TypedDict

import dill
import structlog
from langgraph.graph import END, StateGraph

import directory_parsing as dp
import util
from cfg import Config
from docstring_writing import DocstringWriter, DocstringWritingTool
from feature_analyzing import FeatureAnalyzer, FeatureAnalyzingTool
from file_analyzing import FileAnalyzer, FileAnalyzingTool
from readme_writing import ReadMeWriter, ReadMeWritingTool
from setup_instructing import SetupInstructionTool, SetupInstructor
from usage_instructing import UsageInstructionTool, UsageInstructor
from util import Repository

logger = structlog.get_logger(__name__)


class GraphState(TypedDict):
    user_context: Optional[str] = None
    repository: bytes = None
    feature_summary: str = None
    setup_instructions: str = None
    usage_instructions: str = None
    readme: str = None


class DocumentationWriter:
    def __init__(self, config: Config) -> None:
        util.check_openai_env()

        if config.use_langsmith:
            util.set_langchain_env(config)

        self.config = config
        self.app = self._build_app()

    def _build_app(self) -> StateGraph:
        """
        Constructs a state graph for the documentation writing application.

        This method initializes various components required for the documentation
        writing process, such as docstring writing, file analysis, feature analysis,
        setup instruction, usage instruction, and README writing. It then constructs a
        state graph by adding these components as nodes and defining the sequence of
        operations through directed edges. The state graph is compiled and returned for
        execution.

        Returns
        -------
        StateGraph
            A compiled state graph representing the sequence of operations for the
        documentation writing application.
        """
        docstring_writer = DocstringWriter(
            "docstring_writer.txt", DocstringWritingTool, self.config
        )
        file_analyzer = FileAnalyzer(
            "file_analyzer.txt", FileAnalyzingTool, self.config
        )
        feature_analyzer = FeatureAnalyzer(
            "feature_analyzer.txt", FeatureAnalyzingTool, self.config
        )
        setup_instructor = SetupInstructor(
            "setup_instructor.txt", SetupInstructionTool, self.config
        )
        usage_instructor = UsageInstructor(
            "usage_instructor.txt", UsageInstructionTool, self.config
        )
        readme_writer = ReadMeWriter(
            "readme_writer.txt", ReadMeWritingTool, self.config
        )

        builder = StateGraph(GraphState)

        builder.add_node("docstring_writer", docstring_writer)
        builder.add_node("file_analyzer", file_analyzer)
        builder.add_node("feature_analyzer", feature_analyzer)
        builder.add_node("setup_instructor", setup_instructor)
        builder.add_node("usage_instructor", usage_instructor)
        builder.add_node("readme_writer", readme_writer)

        builder.set_entry_point("docstring_writer")
        builder.add_edge("docstring_writer", "file_analyzer")
        builder.add_edge("file_analyzer", "feature_analyzer")
        builder.add_edge("feature_analyzer", "setup_instructor")
        builder.add_edge("setup_instructor", "usage_instructor")
        builder.add_edge("usage_instructor", "readme_writer")
        builder.add_edge("readme_writer", END)

        return builder.compile()

    def run_app(self, repository: Repository, config: Config) -> None:
        """
        Executes the main application logic for processing a repository based on the
        provided configuration.

        This method checks the configuration to determine whether to modify existing
        files or create new ones. If the configuration specifies modifying existing
        files and the repository is not under version control, it prompts the user for
        confirmation. It then invokes the application logic using the provided
        repository and configuration, processes the final state, and either constructs
        new directories or modifies existing files based on the configuration.

        Parameters
        ----------
        repository : Repository
            The repository object containing the directory tree and call graph to be
        processed.
        config : Config
            The configuration object specifying user context, directory paths, and
        operation modes.

        Returns
        -------
        None
        """
        if (
            config.return_mode == "modify_existing"
            and not repository.directory_tree.git_in_use
        ):
            prompt = (
                "'.git' file not detected in the directory. Are you sure you want to "
                "let the DocumentationWriter modify existing Python files in place? "
                "[Y/n] "
            )
            resp = input(prompt) or "y"
            if resp.lower() not in ["y", "yes"]:
                logger.info("App run canceled.")
                return

        inputs = {
            "user_context": config.user_context,
            "repository": dill.dumps(repository)
        }
        final_state = self.app.invoke(inputs)
        final_repository = dill.loads(final_state["repository"])
        readme = final_state.get("readme")
        if config.return_mode == "create_new":
            dp.construct_new_py_directories(
                final_repository.directory_tree, readme, config.new_dir_location
            )
        elif config.return_mode == "modify_existing":
            dp.modify_existing_py_files(
                final_repository.directory_tree, readme, config.directory
            )
