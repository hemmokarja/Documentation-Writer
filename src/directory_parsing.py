import os
from collections import deque
from pathlib import Path
from typing import Iterator, List, Tuple, Union

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


def _is_py_file(name: str) -> bool:
    """
    Checks if a given filename has a Python file extension.

    Parameters
    ----------
    name : str
        The name of the file to check.

    Returns
    -------
    bool
        True if the file name ends with '.py', indicating it is a Python file;
    otherwise, False.
    """
    return name.endswith(".py")


def _is_text_file(name: str) -> bool:
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


def _is_license(name: str) -> bool:
    """
    Determines if a given name corresponds to a license file.

    Parameters
    ----------
    name : str
        The name to check against the standard license file name.

    Returns
    -------
    bool
        True if the name is 'LICENSE', otherwise False.
    """
    return name == "LICENSE"


class FileNode:
    def __init__(self, path: str, content: str) -> None:
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

    def add_file_summary(self, summary: str) -> None:
        """
        Assigns a summary to the FileNode instance.

        This method sets the `summary` attribute of the FileNode instance to the
        provided string, ensuring that the input is of the correct type.

        Parameters
        ----------
        summary : str
            A string representing the summary to be associated with the file.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the provided summary is not of type `str`.
        """
        if not isinstance(summary, str):
            raise RuntimeError("FileNode summary must be of type `str`")
        self.summary = summary

    def mark_whether_setup_file(self, is_setup_file: bool) -> None:
        """
        Marks the file as a setup file based on the provided input.

        Parameters
        ----------
        is_setup_file : bool or str
            A boolean or string indicating whether the file is a setup file. If a string
        is provided, it is converted to a boolean by checking if it matches common true
        values ('true', 't', 'yes', 'y').

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if isinstance(is_setup_file, bool):
            self.is_setup_file = is_setup_file
        elif isinstance(is_setup_file, str):
            # shouldn't be str but check for good measure
            self.is_setup_file = (
                is_setup_file.lower() in ["true", "t", "yes", "y"]
            )

    def mark_whether_entrypoint_file(self, is_entrypoint_file: bool) -> None:
        """
        Marks the file as an entry point file based on the provided input.

        Parameters
        ----------
        is_entrypoint_file : bool or str
            A boolean or string indicating whether the file is an entry point. If a
        string is provided, it is converted to a boolean by checking if it matches
        common true values ('true', 't', 'yes', 'y').

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if isinstance(is_entrypoint_file, bool):
            self.is_entrypoint_file = is_entrypoint_file
        elif isinstance(is_entrypoint_file, str):
            self.is_entrypoint_file = (
                is_entrypoint_file.lower() in ["true", "t", "yes", "y"]
            )


class DirectoryNode:
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.basename(path)
        self.children = []

    def __repr__(self):
        return f"DirectoryNode(path={self.path})"

    def add_child_to_directory(self, node: Union[FileNode, "DirectoryNode"]) -> None:
        """
        Adds a child node to the directory's list of children.

        This method appends a given node, which can be either a FileNode or another
        DirectoryNode, to the children list of the current DirectoryNode instance. This
        is used to build a hierarchical structure of directories and files.

        Parameters
        ----------
        node : Union[FileNode, DirectoryNode]
            The node to be added as a child to the current directory. It can be a file
        or another directory.
        """
        self.children.append(node)


class DirectoryTree:
    def __init__(self) -> None:
        self.root_directory = None
        self.root_node = None
        self.git_in_use = False

    def __repr__(self):
        return f"DirectoryTree(dir={self.root_directory})"

    def _parse_directory(self, path: str) -> Union[DirectoryNode, None]:
        """
        Recursively parses a directory and constructs a tree of DirectoryNode and
        FileNode objects.

        This method traverses the directory specified by the given path, creating a
        DirectoryNode for each directory and a FileNode for each file. It skips
        directories and files specified in the SKIP_DIRS, SKIP_FILES, and
        SKIP_EXTENSIONS lists. If a directory is not accessible due to permissions, it
        returns None. The method also checks for the presence of a '.git' directory to
        set the git_in_use flag.

        Parameters
        ----------
        path : str
            The path to the directory to be parsed.

        Returns
        -------
        Union[DirectoryNode, None]
            A DirectoryNode representing the parsed directory and its contents, or None
        if the directory is not accessible or does not exist.
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

    def parse_tree(self, root_directory: str) -> None:
        """
        Parses the directory tree starting from the specified root directory.

        This method initializes the directory tree parsing process by setting the root
        directory and invoking the internal method `_parse_directory` to construct a
        tree of `DirectoryNode` and `FileNode` objects. The resulting tree structure is
        stored in the `root_node` attribute of the `DirectoryTree` instance.

        Parameters
        ----------
        root_directory : str
            The path to the root directory from which the directory tree parsing begins.
        """
        self.root_directory = root_directory
        self.root_node = self._parse_directory(self.root_directory)

    def walk(self) -> Iterator[Tuple[str, List[DirectoryNode], List[FileNode]]]:
        """
        Traverses the directory tree starting from the root node and yields information
        about each directory.

        This method performs a breadth-first traversal of the directory tree, starting
        from the root node. For each directory encountered, it yields a tuple containing
        the directory's path, a list of its subdirectories, and a list of its files. The
        traversal continues until all directories in the tree have been visited.

        Returns
        -------
        Iterator[Tuple[str, List[DirectoryNode], List[FileNode]]]
            An iterator that yields tuples, each containing the path of a directory, a
        list of its subdirectory nodes, and a list of its file nodes.
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


def construct_new_py_directories(
    directory_tree: DirectoryTree, readme: str, path: str
) -> None:
    """
    Constructs a new Python project directory structure based on a given directory tree.

    This function takes a directory tree representation and a README content string,
    then creates the corresponding directory and file structure on the filesystem at the
    specified path. It ensures that all directories and files are created as per the
    structure defined in the directory tree. If the directory tree does not have a root
    node, it raises a RuntimeError.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree object representing the structure to be created.
    readme : str
        The content to be written into a README.md file in the root of the new directory
    structure.
    path : str
        The filesystem path where the new directory structure should be created.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the directory tree does not have a root node, indicating that the tree has
    not been parsed.
    """

    def _create_directory(
        node: Union[DirectoryNode, FileNode], parent_path: str
    ) -> None:
        """
        Recursively creates directories and files based on the given node structure.

        This function traverses a node structure, which can consist of directories and
        files, and creates the corresponding directories and files on the filesystem. If
        the node is a directory, it creates the directory and recursively processes its
        children. If the node is a file, it creates the file and writes its content.

        Parameters
        ----------
        node : Union[DirectoryNode, FileNode]
            The node representing either a directory or a file to be created.
        parent_path : str
            The path to the parent directory where the node should be created.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If an error occurs while creating directories or files on the filesystem.
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


def modify_existing_py_files(
    directory_tree: DirectoryTree, readme: str, path: str
) -> None:
    """
    Modifies existing Python files in a directory tree and updates the README file.

    This function traverses a given directory tree, modifying the content of each Python
    file by overwriting it with the content stored in the corresponding file node. It
    also updates the README file at the specified path with the provided content. If the
    directory tree does not have a root node, a RuntimeError is raised.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree containing nodes representing directories and files to be
    modified.
    readme : str
        The content to be written to the README.md file.
    path : str
        The path where the README.md file is located.

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

    def _modify_files(node: Union[DirectoryNode, FileNode]) -> None:
        """
        Recursively modifies Python files within a directory tree node.

        This function traverses a directory tree starting from the given node. If the
        node is a directory, it recursively processes each child node. If the node is a
        Python file, it attempts to open the file in write mode and overwrite its
        content with the content stored in the node. If an error occurs during file
        modification, an error message is printed, and the exception is raised.

        Parameters
        ----------
        node : Union[DirectoryNode, FileNode]
            The node representing either a directory or a file. If it is a directory,
        its children will be processed recursively. If it is a file, it will be modified
        if it is a Python file.

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
