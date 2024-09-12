# Multi-Agent Documentation Generator for Python Repositories

## (Human-generated) Forewords

Code documentation is often viewed as one of the more tedious tasks in programming, yet it's crucial for maintaining a clean, understandable, and sustainable codebase. To ease this burden—both for myself and maybe for others, too—I developed this LLM-powered multi-agent system designed to automate the creation of documentation, specifically, the docstrings and the README.

As proof of the system’s capabilities, the remainder of this repository’s documentation, including the docstrings and this README, has been generated entirely by the AI system.

## Overview
This project is a multi-agent application designed to automate the documentation of Python-based repositories using Large Language Models (LLMs). The system analyzes a given repository by parsing its codebase and generating a function call graph, which outlines the relationships and dependencies between various functions. The application performs two key tasks:

1. **Automatic Docstring Generation**: For each function in the repository, the agents collaborate to write clear, informative docstrings that describe the function's purpose, inputs, outputs, and any other relevant details.
2. **README File Creation**: The system automatically drafts a comprehensive README file that provides an overview of the repository, including its purpose, key features, installation instructions, usage guidelines, and any dependencies.

By leveraging a multi-agent architecture, the application ensures a collaborative approach to analyzing code structure, writing documentation, and providing clear, human-readable insights into the repository. This tool streamlines the process of creating high-quality, standardized documentation for Python projects, making it easier for developers to maintain and share their work.

## Features
- **Automatic Docstring Generation**: Managed by the `docstring_writing.py` file, this feature uses the `DocstringWritingTool` and `DocstringWriter` classes to traverse the function call graph and generate docstrings for each function node.
- **README File Creation**: Distributed across several files, this feature includes:
  - `setup_instructing.py` for generating 'Installation' and 'Environment Setup' sections.
  - `usage_instructing.py` for generating the 'Example Usage' section.
  - `feature_analyzing.py` for analyzing the core features of the repository.
  - `readme_writing.py` for compiling various sections into a structured README file.
- **File and Directory Parsing**: Implemented in `directory_parsing.py`, this feature builds a hierarchical representation of the project.
- **Call Graph Parsing**: Implemented in `call_graph_parsing.py`, this feature constructs a function call graph by parsing Python code.
- **Configuration Management**: Managed by `cfg.py`, which defines a `Config` data class.
- **Utility Functions**: Provided by `util.py` to support environment setup and tool interactions.

## Setup

To set up the project, follow these steps to install the necessary dependencies and configure your environment.

### Prerequisites

- Ensure you have Python 3.9 installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).
- Install Poetry, a tool for dependency management and packaging in Python. You can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hemmokarja/Documentation-Writer.git
   cd Documentation-Writer
   ```

2. **Install dependencies**:
   
   Use Poetry to install the project dependencies specified in the `pyproject.toml` file:
   ```bash
   poetry install
   ```
   This command will create a virtual environment and install all the required packages, including both main and development dependencies.

3. **Activate the virtual environment**:
   
   Once the installation is complete, activate the virtual environment created by Poetry:
   ```bash
   poetry shell
   ```

### Environment Setup

- The project does not specify additional environment variables or configurations in the provided `pyproject.toml`. However, ensure that your environment is correctly set up to use the installed packages.

## Usage Instructions

After completing the setup and installation steps, you can use the project to automatically generate documentation for a Python repository. The main entry point for this application is the `main.py` script located in the `Documentation-Writer/src` directory.

### Running the Documentation Generator

To run the documentation generator, use the following command in your terminal:

```bash
python src/main.py --directory <path-to-your-python-project>
```

#### Parameters:
- `--directory`: (Required) Specify the path to the directory of the Python project you want to document.
- `--user_context`: (Optional) Provide a high-level context of the repository. If not specified, a default context is used.
- `--return_mode`: (Optional) Choose between `create_new` (default) to create a new directory with the documentation or `modify_existing` to update the existing files.
- `--new_dir_location`: (Optional) Specify the location for the new directory if `create_new` mode is used. Default is `document-writer-result-dir`.

#### Example:

```bash
python src/main.py --directory /path/to/my-python-project --return_mode create_new
```

This command will analyze the specified Python project, generate docstrings for each function, and create a new directory with the updated files and a comprehensive README file.

#### Output:
- A new directory (or modified existing directory) containing Python files with generated docstrings and a README file summarizing the project.

## License

This project is licensed under the CC BY-NC 4.0 License. See the LICENSE file for details.