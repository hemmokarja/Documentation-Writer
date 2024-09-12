from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    directory: str = None
    user_context: Optional[str] = None
    return_mode: str = field(default="create_new")
    new_dir_location: str = "documentation-writer-result-dir"
    model_name: str = "gpt-4o-2024-08-06"
    temperature: int = 0
    max_width: int = 88
    indent_size: int = 4
    call_chain_context_up: int = 2  # for docstring writer
    call_chain_context_down: int = 2  # for docstring writer
    use_langsmith: bool = True
    langchain_project: Optional[str] = "documentation-writer"
    langchain_tracing_v2: Optional[str] = "true"
    langchain_endpoint: Optional[str] = "https://api.smith.langchain.com"
    langchain_user_agent: Optional[str] = "myagent"

    def __post_init__(self):
        """
        Validates the configuration settings after initialization.

        This method checks the consistency of the configuration settings, specifically
        ensuring that if the `return_mode` is set to 'create_new', the
        `new_dir_location` must not be None. If this condition is not met, a
        `ValueError` is raised.

        Raises
        ------
        ValueError
            If `return_mode` is 'create_new' and `new_dir_location` is None."""
        if self.return_mode == "create_new" and self.new_dir_location is None:
            raise ValueError(
                "`new_dir_location` cannot be None when `return_mode` is 'create_new'"
            )
