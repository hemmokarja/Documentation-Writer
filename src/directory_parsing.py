import os
from collections import deque
from pathlib import Path

SKIP_DIRS = [
    ".venv",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".idea",
    ".vscode"
]

SKIP_FILES = [
    "__init__.py",
    "poetry.lock"
]

SKIP_EXTENSIONS = [
    ".ipynb",
    ".log",
    ".bak"
]


def _parse_module_name(directory, filepath):
    """
    Parses the module name from a given file path relative to a specified directory.

    Parameters
    ----------
    directory : str or Path
        The base directory from which the file path is relative.
    filepath : str or Path
        The full path to the Python file for which the module name is to be determined.

    Returns
    -------
    str
        The module name derived from the file path, formatted as a Python module path
    with dots instead of slashes and without the '.py' extension.

    Raises
    ------
    ValueError
        If the filepath is not relative to the directory.
    """
    # TODO käytä tai poista
    filepath = Path(filepath)
    directory = Path(directory)
    module_path = str(filepath.relative_to(directory))
    module_name = module_path.replace(".py", "").replace("/", ".")
    return module_name


def _is_py_file(name):
    """
    Checks if a given filename corresponds to a Python file.

    Parameters
    ----------
    name : str
        The name of the file to check.

    Returns
    -------
    bool
        True if the file name ends with the '.py' extension, indicating it is a Python
    file; otherwise, False.
    """
    return name.endswith(".py")


def _is_text_file(name):
    """
    Determines if a given file name corresponds to a text file based on its extension.

    Parameters
    ----------
    name : str
        The name of the file to check.

    Returns
    -------
    bool
        True if the file name ends with the '.txt' extension, indicating it is a text
    file; otherwise, False.
    """
    return name.endswith(".txt")


def _is_license(name):
    """
    Checks if the given name matches the string 'LICENSE'.

    Parameters
    ----------
    name : str
        The name to be checked against the string 'LICENSE'.

    Returns
    -------
    bool
        True if the name is 'LICENSE', otherwise False.
    """
    return name == "LICENSE"


class FileNode:
    def __init__(self, path, content):
        self.path = path
        self.content = content
        self.name = os.path.basename(path)
        self.is_py_file = _is_py_file(self.name)
        self.is_text_file = _is_text_file(self.name)
        self.is_license = _is_license(self.name)
        self.is_setup_file = False  # determined by FileAnalyzer
        self.is_entrypoint_file = False  # determined by FileAnalyzer
        self.summary = None

    def __repr__(self):
        return f"FileNode(path={self.path})"

    def add_file_summary(self, summary):
        """
        Assigns a summary to the FileNode instance.

        Parameters
        ----------
        summary : str
            A string representing the summary of the file.

        Raises
        ------
        RuntimeError
            If the provided summary is not of type `str`.
        """
        if not isinstance(summary, str):
            raise RuntimeError("FileNode summary must be of type `str`")
        self.summary = summary

    def mark_whether_setup_file(self, is_setup_file):
        """
        Marks the file as a setup file based on the provided input.

        Parameters
        ----------
        is_setup_file : bool or str
            A boolean or string indicating whether the file is a setup file. If a string
        is provided, it is converted to a boolean by checking if it matches common true
        values ('true', 't', 'yes', 'y').

        Raises
        ------
        TypeError
            If `is_setup_file` is neither a boolean nor a string.
        """
        if isinstance(is_setup_file, bool):
            self.is_setup_file = is_setup_file
        elif isinstance(is_setup_file, str):
            self.is_setup_file = (
                is_setup_file.lower() in ["true", "t", "yes", "y"]
            )

    def mark_whether_entrypoint_file(self, is_entrypoint_file):
        """
        Marks the file as an entry point file based on the provided input.

        Parameters
        ----------
        is_entrypoint_file : bool or str
            A boolean or string indicating whether the file is an entry point. If a
        string is provided, it is converted to a boolean by checking if it matches
        common true values ('true', 't', 'yes', 'y').

        Raises
        ------
        TypeError
            If `is_entrypoint_file` is neither a boolean nor a string.
        """
        if isinstance(is_entrypoint_file, bool):
            self.is_entrypoint_file = is_entrypoint_file
        elif isinstance(is_entrypoint_file, str):
            self.is_entrypoint_file = (
                is_entrypoint_file.lower() in ["true", "t", "yes", "y"]
            )


class DirectoryNode:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.children = []

    def __repr__(self):
        return f"DirectoryNode(path={self.path})"

    def add_child_to_directory(self, node):
        """
        Adds a child node to the directory node's list of children.

        Parameters
        ----------
        node : DirectoryNode or FileNode
            The node to be added as a child to the current directory node.
        """
        self.children.append(node)


class DirectoryTree:
    def __init__(self):
        self.root_directory = None
        self.root_node = None
        self.git_in_use = False

    def __repr__(self):
        return f"DirectoryTree(dir={self.root_directory})"

    def _parse_directory(self, path):
        """
        Recursively parses a directory and constructs a tree of directory and file
        nodes.

        This method traverses the directory specified by the given path, creating a
        DirectoryNode for each directory and a FileNode for each file. It skips
        directories
        and files specified in the SKIP_DIRS, SKIP_FILES, and SKIP_EXTENSIONS lists. If
        a
        subdirectory is encountered, the method is called recursively to parse it. Files
        are
        read and their contents are stored in FileNode objects. The method returns a
        DirectoryNode representing the root of the parsed directory tree.

        Parameters
        ----------
        path : str
            The path to the directory to be parsed.

        Returns
        -------
        DirectoryNode or None
            A DirectoryNode representing the parsed directory tree, or None if the path
        is
            not a directory or if a PermissionError is encountered.
        """
        if not os.path.isdir(path):
            return None

        directory_node = DirectoryNode(path)

        try:
            entries = os.listdir(path)
        except PermissionError:
            return None

        for entry in entries:

            if entry == ".git":
                self.git_in_use

            if (
                entry in SKIP_DIRS
                or entry in SKIP_FILES
                or any(entry.endswith(ext) for ext in SKIP_EXTENSIONS)
            ):
                continue

            entry_path = os.path.join(path, entry)

            if os.path.isdir(entry_path):
                subdir_node = self._parse_directory(entry_path)
                if subdir_node and subdir_node.children:
                    directory_node.add_child_to_directory(subdir_node)

            elif os.path.isfile(entry_path):
                with open(entry_path, "r") as f:
                    content = f.read()
                file_node = FileNode(entry_path, content)
                directory_node.add_child_to_directory(file_node)

        return directory_node

    def parse_tree(self, root_directory):
        """
        Parses the directory tree starting from the specified root directory and
        initializes the root node of the tree.

        This method sets the root directory of the DirectoryTree instance and uses the
        _parse_directory method to recursively parse the directory structure, creating a
        tree of DirectoryNode and FileNode objects. The root node of this tree is stored
        in the root_node attribute.

        Parameters
        ----------
        root_directory : str
            The path to the root directory from which to start parsing the directory
        tree.
        """
        self.root_directory = root_directory
        self.root_node = self._parse_directory(self.root_directory)

    def walk(self):
        """
        Traverses the directory tree starting from the root node and yields the path,
        subdirectories, and files for each directory.

        This method performs a breadth-first traversal of the directory tree, using a
        queue to manage the nodes to be visited. For each directory node, it separates
        its children into subdirectories and files, then yields the directory's path
        along with lists of its subdirectories and files.

        Yields
        ------
        tuple
            A tuple containing the path of the current directory (str), a list of
        subdirectory nodes (list of DirectoryNode), and a list of file nodes (list of
        FileNode).
        """
        if not self.root_node:
            return

        queue = deque([self.root_node])

        while queue:
            current_dir = queue.popleft()

            files = []
            subdirs = []

            for child in current_dir.children:
                if isinstance(child, DirectoryNode):
                    subdirs.append(child)
                elif isinstance(child, FileNode):
                    files.append(child)

            yield current_dir.path, [subdir for subdir in subdirs], files

            queue.extend(subdirs)


def construct_new_py_directories(directory_tree, readme, path):
    """
    Constructs a new Python project directory structure based on a given directory tree.

    This function takes a directory tree representation and creates the corresponding
    directory and file structure on the filesystem. It uses a helper function to
    recursively create directories and files. Additionally, it writes a README file at
    the root of the newly created directory structure.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree object representing the structure to be created.
    readme : str
        The content to be written into the README.md file at the root of the directory.
    path : str
        The root path where the new directory structure should be created.

    Raises
    ------
    RuntimeError
        If the directory tree does not have a root node, indicating that it has not been
    properly parsed or initialized.
    """

    def _create_directory(node, parent_path):
        """
        Recursively creates a directory structure and files based on the given node.

        This function checks if the provided node is a directory or a file. If it is a
        directory, it creates the directory at the specified parent path and recursively
        processes its children. If it is a file, it creates the file at the specified
        path and writes its content.

        Parameters
        ----------
        node : DirectoryNode or FileNode
            The node representing either a directory or a file to be created.
        parent_path : str
            The path where the directory or file should be created.

        Raises
        ------
        OSError
            If the directory or file cannot be created due to system-related errors.
        """
        if isinstance(node, DirectoryNode):
            current_path = os.path.join(parent_path, node.name)
            os.makedirs(current_path, exist_ok=True)

            for child in node.children:
                _create_directory(child, current_path)
        elif isinstance(node, FileNode):
            filepath = os.path.join(parent_path, node.name)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(node.content)

    if directory_tree.root_node:
        _create_directory(directory_tree.root_node, path)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(readme)
    else:
        raise RuntimeError(
            "Cannot reconstruct directory structure without root node. Please, "
            "parse the directory tree by executing `DirectoryTree.parse_tree()`"
        )


def modify_existing_py_files(directory_tree, readme, path):
    """
    Modifies existing Python files in a directory tree and updates the README file.

    This function traverses a given directory tree, modifying the content of each Python
    file by overwriting it with the content stored in the corresponding node. It also
    updates the README file at the specified path with the provided content. If the
    directory tree does not have a root node, a RuntimeError is raised.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree containing nodes representing directories and files.
    readme : str
        The content to be written to the README.md file.
    path : str
        The path where the README.md file should be updated.

    Raises
    ------
    RuntimeError
        If the directory tree does not have a root node, indicating that it has not been
    parsed.
    IOError
        If an I/O error occurs during file modification.
    OSError
        If an OS-related error occurs during file modification.
    """

    def _modify_files(node):
        """
        Recursively modifies Python files within a directory tree node.

        This function traverses a directory tree starting from the given node. If the
        node is a directory, it recursively processes each child node. If the node is a
        Python file, it attempts to open the file in write mode and overwrite its
        content with the content stored in the node. If an error occurs during file
        modification, it prints an error message and raises the exception.

        Parameters
        ----------
        node : DirectoryNode or FileNode
            The node representing either a directory or a file in the directory tree. If
        it is a directory, its children are recursively processed. If it is a file, it
        is modified if it is a Python file.

        Raises
        ------
        IOError
            If an I/O error occurs during file modification.
        OSError
            If an OS-related error occurs during file modification.
        """
        if isinstance(node, DirectoryNode):
            for child in node.children:
                _modify_files(child)
        elif isinstance(node, FileNode) and node.is_py_file:
            try:
                with open(node.path, "w") as f:
                    f.write(node.content)
            except (IOError, OSError):
                print(f"Error modifying file {node.path}")
                raise

    if directory_tree.root_node:
        _modify_files(directory_tree.root_node)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(readme)
    else:
        raise RuntimeError(
            "Cannot modify files without root node. Please, parse the directory "
            "tree by executing `DirectoryTree.parse_tree()`."
        )
