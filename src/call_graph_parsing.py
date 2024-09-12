import ast
import re
import textwrap
from typing import Iterator, List, Literal, Optional, Set, Tuple

from directory_parsing import DirectoryTree, FileNode
from dunders import DUNDER_METHODS


class ClassNode:
    # container for class definitions shared between several class methods
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def __repr__(self):
        return f"ClassNode(name={self.name})"


class FunctionNode:
    def __init__(
        self,
        name: str,
        lineno: int,
        col_offset: int,
        definition: str,
        docstring: Optional[str] = None,
        class_node: Optional[ClassNode] = None,
        file_node: Optional[FileNode] = None,
    ) -> None:
        self.name = name
        self.lineno = lineno
        self.col_offset = col_offset
        self.definition = definition
        self.docstring = docstring
        self.class_node = class_node
        self.file_node = file_node

    def __repr__(self):
        return (
            f"FunctionNode(name={self.name}, "
            f"location={self.file_node.name}:{self.lineno})"
        )

    def __eq__(self, other):
        return (
            isinstance(other, FunctionNode)
            and self.name == other.name
            and self.file_node.path == other.file_node.path
            and self.lineno == other.lineno
        )

    def __hash__(self):
        return hash((self.name, self.file_node.path, self.lineno))


class FileVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_definitions = set()
        self.class_nodes = {}
        self.function_stack = []  # stack is for handling nested functions
        self.deferred_calls = []
        self.source_code = None
        self.file_node = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visits a function definition node in the abstract syntax tree (AST) and
        processes it.

        This method is responsible for handling function definition nodes within the
        AST. It checks if the function is not a dunder method and processes it
        accordingly. If the function is part of a class, it associates the function with
        its class node. It extracts the function's docstring, if available, and creates
        a FunctionNode object representing the function. This object is then added to
        the function stack and the set of function definitions for further processing.

        Parameters
        ----------
        node : ast.FunctionDef
            The AST node representing a function definition to be visited and processed.

        Returns
        -------
        None
            This method does not return any value; it updates the internal state of the
        FileVisitor instance by adding the function node to the stack and the set of
        function definitions.
        """
        function_name = node.name
        if function_name not in DUNDER_METHODS:

            if isinstance(node.parent, ast.ClassDef):
                class_name = node.parent.name
                class_definition = ast.get_source_segment(
                    self.source_code, node.parent
                )
                if class_name not in self.class_nodes:
                    self.class_nodes[class_name] = ClassNode(
                        class_name, class_definition
                    )
                class_node = self.class_nodes[class_name]
            else:
                class_node = None

            docstring = ast.get_docstring(node)
            if docstring:
                docstring = docstring.replace("\n", " ")

            function_node = FunctionNode(
                name=function_name,
                lineno=node.lineno,
                col_offset=node.col_offset,
                definition=ast.get_source_segment(self.source_code, node, padded=True),
                docstring=docstring,
                class_node=class_node,
                file_node=self.file_node,
            )

            self.function_stack.append(function_node)
            self.function_definitions.add(function_node)

            self.generic_visit(node)
            self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visits a function call node in the abstract syntax tree (AST) and defers its
        processing.

        This method is responsible for handling function call nodes within the AST. If
        there is an active function on the function stack, it identifies the caller
        function and the name of the function being called. It then defers the
        processing of this call by appending it to the deferred_calls list, which will
        be processed after all functions have been visited. This allows for handling
        function calls in the context of their definitions.

        Parameters
        ----------
        node : ast.Call
            The AST node representing a function call to be visited and deferred.

        Returns
        -------
        None
            This method does not return any value; it updates the internal state of the
        FileVisitor instance by adding the call to the deferred_calls list.
        """
        if self.function_stack:
            caller_function = self.function_stack[-1]
            function_name = self._get_function_name(node)
            # defer processing this call until all functions have been visited
            self.deferred_calls.append((caller_function, function_name))

        self.generic_visit(node)

    def _get_function_name(self, node: ast.Call) -> str:
        """
        Extracts the function name from an AST call node.

        This method determines the name of a function being called by examining the
        provided AST call node. It handles both simple function calls and method calls
        on objects, returning the appropriate function or method name.

        Parameters
        ----------
        node : ast.Call
            The AST node representing a function call from which to extract the function
        name.

        Returns
        -------
        str
            The name of the function or method being called.

        Raises
        ------
        RuntimeError
            If the function name cannot be extracted from the node, a RuntimeError is
        raised.
        """
        if isinstance(node.func, ast.Name):
            # simple function call like foo()
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # object method call like class.foo()
            return node.func.attr
        raise RuntimeError("Failed to extract function name from node")

    def _add_parent_references(
        self, node: ast.AST, parent: Optional[ast.AST] = None
    ) -> None:
        """
        Assigns parent references to each node in an abstract syntax tree (AST).

        This method recursively traverses the AST starting from the given node, setting
        the parent attribute for each node to facilitate easier navigation and analysis
        of the tree structure.

        Parameters
        ----------
        node : ast.AST
            The root node of the AST from which to start assigning parent references.
        parent : Optional[ast.AST], optional
            The parent node to assign to the current node, by default None.

        Returns
        -------
        None
            This method does not return any value; it modifies the AST nodes in place.
        """
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._add_parent_references(child, node)

    def parse_visits(self, file_node: FileNode) -> None:
        """
        Parses the abstract syntax tree (AST) of a Python file to identify and process
        function definitions and calls.

        This method takes a FileNode object, extracts its source code, and parses it
        into an AST. It then assigns parent references to each node in the AST to
        facilitate navigation and analysis. Finally, it visits each node in the AST to
        identify function definitions and calls, storing them for further processing.

        Parameters
        ----------
        file_node : FileNode
            The file node containing the source code to be parsed.

        Returns
        -------
        None
            This method does not return any value; it processes the AST in place and
        updates the internal state of the FileVisitor instance.
        """
        self.file_node = file_node
        self.source_code = file_node.content
        tree = ast.parse(self.source_code)
        self._add_parent_references(tree, parent=None)
        self.visit(tree)


def _py_filenodes(directory_tree: DirectoryTree) -> Iterator[FileNode]:
    """
    Yields Python file nodes from a directory tree.

    This function traverses a given directory tree and yields each file node that
    represents a Python file. It utilizes the `walk` method of the `DirectoryTree` class
    to perform a breadth-first traversal, checking each file node to determine if it is
    a Python file.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree to traverse for Python file nodes.

    Returns
    -------
    Iterator[FileNode]
        An iterator over file nodes that are identified as Python files.
    """
    for _, _, file_nodes in directory_tree.walk():
        for node in file_nodes:
            if node.is_py_file:
                yield node


class CallResolver:
    def __init__(self):
        self.node_to_callees = {}
        self.deferred_calls = []

    def _add_function_definition(self, function_node: FunctionNode) -> None:
        """
        Adds a function node to the call graph with no callees.

        This method initializes an entry in the `node_to_callees` dictionary for the
        given `function_node`, associating it with an empty set. This indicates that the
        function node currently has no known callees, which is useful for solitary
        functions that do not call other user-defined functions.

        Parameters
        ----------
        function_node : FunctionNode
            The function node to be added to the call graph with no callees.
        """
        self.node_to_callees[function_node] = set()

    def _add_deferred_call(self, caller_node: FunctionNode, callee_name: str) -> None:
        """
        Adds a deferred function call to the list of deferred calls.

        This method records a function call that cannot be immediately resolved by
        storing a tuple consisting of the caller node and the callee name in the
        `deferred_calls` list. Deferred calls are later resolved to actual function
        nodes in the call graph by the `_resolve_deferred_calls` method.

        Parameters
        ----------
        caller_node : FunctionNode
            The function node representing the caller of the deferred call.
        callee_name : str
            The name of the callee function that is being called by the caller node.
        """
        call_signature = (caller_node, callee_name)
        self.deferred_calls.append(call_signature)

    def _resolve_deferred_calls(self) -> None:
        """
        Resolves deferred function calls by matching caller nodes with their
        corresponding callee nodes in the call graph.

        This method iterates over the list of deferred calls, attempting to find a
        matching callee node for each deferred callee name. If a unique match is found,
        the callee node is added to the set of callees for the caller node in the
        `node_to_callees` dictionary. If multiple matches are found, a `RuntimeError` is
        raised, indicating non-unique function names. If no match is found, the deferred
        call is ignored, assuming it refers to a non-user-defined function.

        Raises
        ------
        RuntimeError
            If more than one matching node is found for a deferred call or if the caller
        node is not found in the call graph.
        """
        for caller_node, deferred_callee_name in self.deferred_calls:

            # find the node matching the name of the called function
            matching_callees = [
                node for node in self.node_to_callees
                if node.name == deferred_callee_name
            ]

            if len(matching_callees) > 1:
                raise RuntimeError(
                    f"Encountered more than one matching node candidate for deferred "
                    f"call '{deferred_callee_name}' (called by {caller_node.name}). "
                    "This may happen if function names are not globally unique. "
                    f"Found callee nodes: {matching_callees}"
                )
            if len(matching_callees) == 1:
                callee_node = matching_callees[0]
                if caller_node in self.node_to_callees:
                    self.node_to_callees[caller_node].add(callee_node)
                else:
                    raise RuntimeError(
                        f"Caller node {caller_node} not found from call graph when "
                        "resolving calls."
                    )
            else:
                # non-user defined functions get skipped
                pass

    def resolve_calls(self, directory_tree: DirectoryTree) -> None:
        """
        Resolves function calls within a directory tree by building a call graph.

        This method processes each Python file in the given directory tree to identify
        function definitions and deferred calls. It uses a `FileVisitor` to parse each
        file's abstract syntax tree (AST) and collect function definitions and calls
        that cannot be immediately resolved. These deferred calls are stored for later
        resolution. After processing all files, the method resolves these deferred calls
        by matching them with their corresponding function definitions, updating the
        call graph accordingly.

        Parameters
        ----------
        directory_tree : DirectoryTree
            The directory tree containing Python files to be analyzed for function
        calls.

        Returns
        -------
        None
            This method does not return any value; it updates the internal state of the
        `CallResolver` instance by populating the call graph with function nodes and
        resolving deferred calls.
        """
        for file_node in _py_filenodes(directory_tree):

            visitor = FileVisitor()
            visitor.parse_visits(file_node)

            for node in visitor.function_definitions:
                self._add_function_definition(node)

            for caller_node, callee_name in visitor.deferred_calls:
                self._add_deferred_call(caller_node, callee_name)

        self._resolve_deferred_calls()

    def __repr__(self):
        return f"CallResolver({len(self.node_to_callees)} nodes)"


def _wrap(paragraph: str, width: int, indent_str: int) -> str:
    """
    Wraps a given paragraph to a specified width with a specified indentation.

    Parameters
    ----------
    paragraph : str
        The paragraph of text to be wrapped.
    width : int
        The maximum line width for the wrapped text.
    indent_str : int
        The string used for indentation at the beginning of each line.

    Returns
    -------
    str
        The wrapped paragraph with the specified width and indentation."""
    return textwrap.fill(
        paragraph, width=width, initial_indent=indent_str, subsequent_indent=indent_str
    )


def _ensure_triple_quotes(docstring: str) -> str:
    """
    Ensures that a given docstring is enclosed in triple quotes.

    This function checks if the provided docstring starts and ends with triple quotes,
    either single or double. If not, it prepends and appends triple double quotes to the
    docstring, ensuring it is properly formatted.

    Parameters
    ----------
    docstring : str
        The docstring to be checked and potentially modified.

    Returns
    -------
    str
        The docstring enclosed in triple quotes if it was not already.
    """
    singles = "'''"
    doubles = '"""'

    if not (docstring.startswith(doubles) or docstring.startswith(singles)):
        if not docstring.startswith("\n"):
            docstring = "\n" + docstring
        docstring = doubles + docstring

    if not (docstring.endswith(doubles) or docstring.endswith(singles)):
        if not docstring.endswith("\n"):
            docstring += "\n"
        docstring += doubles

    return docstring


class CallGraphNode(FunctionNode):
    def __init__(
        self,
        name: str,
        lineno: int,
        col_offset: int,
        definition: Optional[str] = None,
        docstring: Optional[str] = None,
        class_node: Optional[ClassNode] = None,
        file_node: Optional[FileNode] = None,
    ) -> None:
        super().__init__(
            name=name,
            lineno=lineno,
            col_offset=col_offset,
            definition=definition,
            docstring=docstring,
            class_node=class_node,
            file_node=file_node
        )
        self.new_docstring = None
        self.summary = None
        self.depth = None
        self.children = []
        self.parents = []

    def __repr__(self):
        return (
            f"CallGraphNode(name={self.name}, "
            f"location={self.file_node.name}:{self.lineno}, "
            f"depth={self.depth})"
        )

    @classmethod
    def from_function_node(cls, function_node: FunctionNode) -> "CallGraphNode":
        """
        Creates a `CallGraphNode` instance from a given `FunctionNode`.

        This class method takes a `FunctionNode` object and initializes a
        `CallGraphNode` with the same attributes, effectively converting the function
        node into a call graph node.

        Parameters
        ----------
        function_node : FunctionNode
            The function node from which to create the call graph node.

        Returns
        -------
        CallGraphNode
            A new instance of `CallGraphNode` initialized with the attributes of the
        given `FunctionNode`."""
        return cls(
            name=function_node.name,
            lineno=function_node.lineno,
            col_offset=function_node.col_offset,
            definition=function_node.definition,
            docstring=function_node.docstring,
            class_node=function_node.class_node,
            file_node=function_node.file_node,
        )

    def add_child(self, child_node: "CallGraphNode") -> None:
        """
        Adds a child node to the current `CallGraphNode` and updates the parent list of
        the child node.

        This method establishes a bidirectional relationship between the current node
        and the specified child node by appending the child node to the current node's
        children list and the current node to the child node's parents list.

        Parameters
        ----------
        child_node : CallGraphNode
            The child node to be added to the current node's children list.
        """
        self.children.append(child_node)
        child_node.parents.append(self)

    def add_function_summary(self, summary: str) -> None:
        """
        Assigns a summary to the `CallGraphNode` instance.

        This method sets the `summary` attribute of the `CallGraphNode` to the provided
        string, which serves as a brief description or overview of the node's purpose or
        functionality.

        Parameters
        ----------
        summary : str
            A string representing the summary to be assigned to the node.

        Raises
        ------
        RuntimeError
            If the provided `summary` is not of type `str`.
        """
        if not isinstance(summary, str):
            raise RuntimeError("CallGraphNode must be of type `str`")
        self.summary = summary

    def add_new_docstring(self, new_docstring: str) -> None:
        """
        Assigns a new docstring to the `CallGraphNode` instance after validating its
        type and ensuring it is properly formatted.

        This method sets the `new_docstring` attribute of the `CallGraphNode` to the
        provided string, ensuring that it is a valid string and is enclosed in triple
        quotes. It raises an error if the provided docstring is not a string.

        Parameters
        ----------
        new_docstring : str
            The new docstring to be assigned to the node. It must be a string.

        Raises
        ------
        RuntimeError
            If the provided `new_docstring` is not of type `str`.
        """
        if not isinstance(new_docstring, str):
            raise RuntimeError("New docstring must be of type `str`")
        self.new_docstring = _ensure_triple_quotes(textwrap.dedent(new_docstring))

    def insert_new_docstring(self, max_width: int, indent_size: int) -> None:
        """
        Inserts a new docstring into the function, class, or file definition associated
        with this `CallGraphNode`.

        This method iterates over possible insertion targets ('function', 'class',
        'file') and calls a helper method to insert the new docstring into the
        appropriate location. If the target is 'class' and the `CallGraphNode` does not
        have an associated class node, the insertion is skipped for that target.

        Parameters
        ----------
        max_width : int
            The maximum width for each line of the new docstring.
        indent_size : int
            The number of spaces to use for indentation of the new docstring.
        """
        for insert_into in ["function", "class", "file"]:
            if insert_into == "class" and not self.class_node:
                continue
            self._insert_new_docstring(max_width, indent_size, insert_into)

    def _insert_new_docstring(
        self,
        max_width: int,
        indent_size: int,
        into: Literal["function", "class", "file"]
    ) -> None:
        """
        Inserts a new docstring into the specified code segment of a function, class, or
        file.

        This method takes a new docstring, formats it according to the specified maximum
        width and indentation size, and inserts it into the appropriate location within
        the code of a function, class, or file. It raises an error if the new docstring
        is not provided or if the target location is invalid.

        Parameters
        ----------
        max_width : int
            The maximum width for each line of the new docstring.
        indent_size : int
            The number of spaces to use for indentation of the new docstring.
        into : {'function', 'class', 'file'}
            The target location where the new docstring should be inserted. Must be one
        of 'function', 'class', or 'file'.

        Raises
        ------
        RuntimeError
            If the new docstring is not provided or if the function is not found in the
        code.
        ValueError
            If the `into` parameter is not one of 'function', 'class', or 'file'.
        """
        if self.new_docstring is None:
            raise RuntimeError(
                "Attempted to insert a new docstring into function definition before "
                "providing a new docstring. Please, add a new docstring to the "
                "function node first with method `add_new_docstring`."
            )
        if into not in ["function", "class", "file"]:
            raise ValueError("`into` must be 'function', or 'class', or 'file'")

        if into == "function":
            code = self.definition
        elif into == "class":
            code = self.class_node.definition
        elif into == "file":
            code = self.file_node.content

        # compose new docstring
        indent_str = " " * (indent_size + self.col_offset)
        wrapped_paragraphs = [
            _wrap(paragraph, max_width, indent_str)
            for paragraph in self.new_docstring.splitlines()
        ]
        wrapped_docstring = "\n".join(wrapped_paragraphs)

        # insert docstring into the specific function definition
        docstring_pattern = re.compile(
            # match the function header
            # rf'(def\s+{self.name}\s*\([^)]*\)\s*:)'  # works without return typehints
            rf'(def\s+{self.name}\s*\([^)]*\)\s*(?:->\s*[^\s:][^\n]*?)?:)'
            # optionally match any existing docstring
            r'(\s*"""[\s\S]*?"""|\s*\'\'\'[\s\S]*?\'\'\')?'
        )
        re_match = re.search(docstring_pattern, code)

        if re_match:
            start, end = re_match.span(2) if re_match.group(2) else (None, None)
            if start and end:
                new_code = (
                    code[:start]
                    + "\n"
                    + wrapped_docstring
                    + code[end:]
                )
            else:
                header_end = re_match.end(1)
                new_code = (
                    code[:header_end]
                    + "\n"
                    + wrapped_docstring
                    + code[header_end:]
                )
        else:
            raise RuntimeError(
                f"Function '{self.name}' not found in the provided code:\n{code}"
            )

        if into == "function":
            self.definition = new_code
        elif into == "class":
            self.class_node.definition = new_code
        elif into == "file":
            self.file_node.content = new_code

    def traverse_down(
        self, max_depth: Optional[int] = None, return_start_node: bool = True
    ) -> List[Tuple[int, "CallGraphNode"]]:
        """
        Traverses the call graph downward from the current node, collecting nodes up to
        a specified depth.

        This method performs a depth-first traversal starting from the current
        `CallGraphNode`, visiting each child node recursively. It collects nodes in a
        list along with their respective depths, stopping when the specified maximum
        depth is reached or all nodes are visited.

        Parameters
        ----------
        max_depth : Optional[int], optional
            The maximum depth to traverse. If None, the traversal continues until all
        nodes are visited, by default None.
        return_start_node : bool, optional
            Whether to include the starting node in the result list, by default True.

        Returns
        -------
        List[Tuple[int, CallGraphNode]]
            A list of tuples where each tuple contains the depth of the node and the
        `CallGraphNode` itself.
        """
        visited = set()
        result = []

        def _traverse_down(node: "CallGraphNode", current_depth: int) -> None:
            """
            Recursively traverses the call graph starting from the given node, exploring
            its children nodes downwards.

            Parameters
            ----------
            node : CallGraphNode
                The starting node of the traversal.
            current_depth : int
                The current depth of the traversal from the starting node.

            Returns
            -------
            None
                This function does not return a value but updates the global 'visited'
            set and 'result' list with the traversal data.

            Notes
            -----
            This function is a helper function used within the 'traverse_down' method to
            perform a depth-first traversal of the call graph. It checks if a node has
            already been visited to avoid cycles and respects a maximum depth limit if
            specified. The traversal results are stored in a global list 'result' as
            tuples of depth and node.
            """
            node_name = node.name
            if node_name in visited:
                return
            visited.add(node_name)
            result.append((current_depth, node))

            if max_depth is not None and current_depth >= max_depth:
                return

            for child in node.children:
                _traverse_down(child, current_depth + 1)

        _traverse_down(self, 0)
        if return_start_node:
            return result
        else:
            return result[1:]

    def traverse_up(
        self, max_depth: Optional[int] = None, return_start_node: bool = True
    ) -> List[Tuple[int, "CallGraphNode"]]:
        """
        Traverses the call graph upward from the current node, collecting nodes up to a
        specified depth.

        This method performs a depth-first traversal starting from the current
        `CallGraphNode`, visiting each parent node recursively. It collects nodes in a
        list along with their respective depths, stopping when the specified maximum
        depth is reached or all nodes are visited.

        Parameters
        ----------
        max_depth : Optional[int], optional
            The maximum depth to traverse. If None, the traversal continues until all
        nodes are visited, by default None.
        return_start_node : bool, optional
            Whether to include the starting node in the result list, by default True.

        Returns
        -------
        List[Tuple[int, CallGraphNode]]
            A list of tuples where each tuple contains the depth of the node and the
        `CallGraphNode` itself.
        """
        visited = set()
        result = []

        def _traverse_up(node: "CallGraphNode", current_depth: int) -> None:
            """
            Recursively traverses up the call graph from a given node, recording each
            node and its depth.

            This function is a helper for traversing the call graph upwards, starting
            from the specified node. It records each visited node along with its depth
            in the traversal. The traversal stops if a node has already been visited or
            if the maximum depth is reached.

            Parameters
            ----------
            node : CallGraphNode
                The starting node for the upward traversal.
            current_depth : int
                The current depth of the traversal from the starting node.

            Returns
            -------
            None
                This function does not return a value but modifies the global 'visited'
            set and 'result' list.
            """
            node_name = node.name
            if node_name in visited:
                return
            visited.add(node_name)
            result.append((current_depth, node))

            if max_depth is not None and current_depth >= max_depth:
                return

            for parent in node.parents:
                _traverse_up(parent, current_depth + 1)

        _traverse_up(self, 0)
        if return_start_node:
            return result
        else:
            return result[1:]


class CallGraph:
    def __init__(self) -> None:
        self.nodes = {}

    def __repr__(self):
        return f"CallGraph({len(self.nodes)} nodes)"

    def _add_node(self, function_node: FunctionNode) -> None:
        """
        Adds a function node to the call graph if it does not already exist.

        This method checks if a function node, identified by its name, is already
        present in the call graph's nodes dictionary. If not, it creates a new
        `CallGraphNode` from the given `FunctionNode` and adds it to the graph.

        Parameters
        ----------
        function_node : FunctionNode
            The function node to be added to the call graph.

        Returns
        -------
        CallGraphNode
            The `CallGraphNode` corresponding to the given `FunctionNode`, whether newly
        created or already existing in the graph.
        """
        name = function_node.name
        if name not in self.nodes:
            self.nodes[name] = CallGraphNode.from_function_node(function_node)
        return self.nodes[name]

    def _add_edge(
        self, parent_fn_node: FunctionNode, child_fn_node: FunctionNode
    ) -> None:
        """
        Establishes a parent-child relationship between two function nodes in the call
        graph.

        This method adds both the parent and child function nodes to the call graph if
        they are not already present, and then creates a directed edge from the parent
        node to the child node. This is achieved by adding the child node to the
        parent's list of children, thereby representing the call relationship in the
        graph.

        Parameters
        ----------
        parent_fn_node : FunctionNode
            The function node representing the parent in the call relationship.
        child_fn_node : FunctionNode
            The function node representing the child in the call relationship.

        Returns
        -------
        None
            This method does not return any value but modifies the call graph by adding
        nodes and establishing edges between them.
        """
        parent_node = self._add_node(parent_fn_node)
        child_node = self._add_node(child_fn_node)
        parent_node.add_child(child_node)

    def parse_graph(self, directory_tree: DirectoryTree) -> None:
        """
        Parses a directory tree to build a call graph representing function call
        relationships.

        This method initializes a `CallResolver` to analyze the given directory tree,
        identifying function calls and their relationships. It iterates over the
        resolved call data to add function nodes and edges to the call graph, ensuring
        that even solitary functions (those without callees) are included. Finally, it
        calculates the depth of each node in the graph, which represents the longest
        path from any entry point to the node.

        Parameters
        ----------
        directory_tree : DirectoryTree
            The directory tree containing Python files to be analyzed for building the
        call graph.

        Returns
        -------
        None
            This method does not return any value; it updates the internal state of the
        `CallGraph` instance by populating it with nodes and edges representing function
        call relationships.
        """
        self.resolver = CallResolver()
        self.resolver.resolve_calls(directory_tree)

        for parent_fn_node, children_fn_nodes in self.resolver.node_to_callees.items():
            # add parent node already here to get it recorded even if it doesn't have
            # children, this may happen for solitary functions that do not call other
            # user-defined functions
            self._add_node(parent_fn_node)
            for child_fn_node in children_fn_nodes:
                self._add_edge(parent_fn_node, child_fn_node)

        self._calculate_depths()

    def get_node(self, name: str) -> CallGraphNode:
        """
        Retrieve a node from the call graph by its name.

        This method looks up a `CallGraphNode` in the call graph's nodes dictionary
        using the provided name. If the node is not found, it raises a `KeyError`.

        Parameters
        ----------
        name : str
            The name of the node to retrieve from the call graph.

        Returns
        -------
        CallGraphNode
            The `CallGraphNode` associated with the given name.

        Raises
        ------
        KeyError
            If the call graph does not contain a node with the specified name.
        """
        node = self.nodes.get(name)
        if node is None:
            raise KeyError(f"CallGraph doesn't contain CallGraphNode '{name}'")
        return node

    def get_entrypoints(self) -> List[CallGraphNode]:
        """
        Retrieve all entry point nodes in the call graph.

        An entry point node is defined as a node with no parent nodes, indicating that
        it is not called by any other node in the graph.

        Returns
        -------
        List[CallGraphNode]
            A list of CallGraphNode objects that are entry points in the call graph,
        meaning they have no parent nodes.
        """
        return [node for node in self.nodes.values() if not node.parents]

    def get_endpoints(self) -> List[CallGraphNode]:
        """
        Retrieve all endpoint nodes in the call graph.

        An endpoint node is defined as a node with no child nodes, indicating that it
        does not call any other node in the graph.

        Returns
        -------
        List[CallGraphNode]
            A list of CallGraphNode objects that are endpoints in the call graph,
        meaning they have no child nodes.
        """
        return [node for node in self.nodes.values() if not node.children]

    def _calculate_depths(self) -> None:
        """
        Calculate the depth of each node in the call graph.

        This method initializes the depth of each node to None and then updates the
        depth using a depth-first search (DFS) approach. The depth of a node is defined
        as the longest path from any entry point node to the node itself. Entry point
        nodes are those with no parent nodes.

        Returns
        -------
        None
            This method does not return any value but updates the `depth` attribute of
        each `CallGraphNode` in the `nodes` dictionary of the `CallGraph` class.
        """
        # initialize with None to guarantee fresh start
        for node in self.nodes.values():
            node.depth = None

        def update_depth(
            node: CallGraphNode, depth: int, visited: Set[CallGraphNode]
        ) -> None:
            """
            Updates the depth of a node in a call graph recursively.

            This function performs a depth-first search (DFS) to update the depth of
            each node in the call graph, starting from the given node. It ensures that
            each node is visited only once by maintaining a set of visited nodes. The
            depth of a node is updated only if the current depth is greater than the
            existing depth.

            Parameters
            ----------
            node : CallGraphNode
                The node whose depth is to be updated.
            depth : int
                The current depth level to assign to the node.
            visited : Set[CallGraphNode]
                A set of nodes that have already been visited to prevent revisiting and
            infinite loops.

            Returns
            -------
            None
                This function does not return any value but updates the depth attribute
            of the nodes in the call graph.
            """
            if node in visited:
                return
            if node.depth is None or node.depth < depth:
                node.depth = depth
                visited.add(node)
                for child in node.children:
                    update_depth(child, depth + 1, visited)

        for node in self.get_entrypoints():
            visited = set()  # track visited nodes in each DFS separately
            update_depth(node, 0, visited)
