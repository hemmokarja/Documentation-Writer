import ast
import re
import textwrap

from dunders import DUNDER_METHODS


class FunctionNode:
    def __init__(
        self,
        name,
        lineno,
        col_offset,
        definition,
        docstring=None,
        class_node=None,
        file_node=None,
    ):
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


class ClassNode:
    # container for class definitions shared between several class methods
    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def __repr__(self):
        return f"ClassNode(name={self.name})"


class FileVisitor(ast.NodeVisitor):
    def __init__(self):
        self.function_definitions = set()
        self.class_nodes = {}
        self.function_stack = []  # stack is for handling nested functions
        self.deferred_calls = []
        self.source_code = None
        self.file_node = None

    def visit_FunctionDef(self, node):
        """
        Visits a FunctionDef node in the abstract syntax tree (AST) to collect and store
        information about function definitions.

        This method is automatically invoked when the visitor encounters a FunctionDef
        node, which represents a function definition in the AST. It processes the node
        to extract relevant details such as the function's name, line number, column
        offset, source code segment, and docstring. If the function is part of a class,
        it also associates the function with its class node. The function information is
        encapsulated in a FunctionNode object, which is then added to the function stack
        and the set of function definitions. The method also ensures that nested
        functions are handled correctly by using a stack to manage the current context.

        Parameters
        ----------
        node : ast.FunctionDef
            The AST node representing a function definition to be visited and processed.
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

    def visit_Call(self, node):
        """
        Visits a Call node in the abstract syntax tree (AST) and defers processing of
        the function call.

        This method is automatically invoked when the visitor encounters a Call node,
        which represents a function call in the AST. If there is an active function on
        the function stack, it records the function call by appending it to the
        deferred_calls list, associating it with the current caller function. This
        deferral allows for processing the call after all functions have been visited,
        ensuring that all necessary context is available.

        Parameters
        ----------
        node : ast.Call
            The AST node representing a function call to be visited and deferred for
        later processing.
        """
        if self.function_stack:
            caller_function = self.function_stack[-1]
            function_name = self._get_function_name(node)
            # defer processing this call until all functions have been visited
            self.deferred_calls.append((caller_function, function_name))

        self.generic_visit(node)

    def _get_function_name(self, node):
        """
        Extracts the function name from a given AST Call node.

        This method determines the name of the function being called by examining the
        node's structure. It handles both simple function calls and method calls on
        objects.

        Parameters
        ----------
        node : ast.Call
            The AST node representing a function call from which to extract the function
        name.

        Returns
        -------
        str
            The name of the function being called.

        Raises
        ------
        RuntimeError
            If the function name cannot be extracted from the node.
        """
        if isinstance(node.func, ast.Name):
            # simple function call like foo()
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # object method call like class.foo()
            return node.func.attr
        raise RuntimeError("Failed to extract function name from node")

    def _add_parent_references(self, node, parent=None):
        """
        Recursively adds references to parent nodes in an Abstract Syntax Tree (AST).

        This method traverses the AST starting from the given node and assigns a
        reference to the parent node for each child node. This is useful for later
        processing steps that require knowledge of a node's context within the tree.

        Parameters
        ----------
        node : ast.AST
            The current node in the AST from which to start adding parent references.
        parent : ast.AST, optional
            The parent node of the current node. Defaults to None, indicating that the
        current node is the root of the tree.
        """
        # recursively adds references to parent nodes
        node.parent = parent
        for child in ast.iter_child_nodes(node):
            self._add_parent_references(child, node)

    def parse_visits(self, file_node):
        """
        Parses the abstract syntax tree (AST) of a Python source file and visits each
        node to collect function definitions and deferred function calls.

        This method initializes the `file_node` and `source_code` attributes with the
        provided file node's content. It then parses the source code into an AST and
        adds parent references to each node in the tree. Finally, it visits each node in
        the AST to identify and store function definitions and deferred calls for later
        processing.

        Parameters
        ----------
        file_node : FileNode
            The file node containing the source code to be parsed and visited.
        """
        self.file_node = file_node
        self.source_code = file_node.content
        tree = ast.parse(self.source_code)
        self._add_parent_references(tree, parent=None)
        self.visit(tree)


def _py_filenodes(directory_tree):
    """
    Yields Python file nodes from a directory tree.

    This function traverses a given directory tree and yields nodes that represent
    Python files. It utilizes the `walk` method of the directory tree to iterate over
    all files and directories, filtering out only those files that are identified as
    Python files.

    Parameters
    ----------
    directory_tree : DirectoryTree
        The directory tree to traverse, which provides a `walk` method to iterate over
    its nodes.

    Yields
    ------
    FileNode
        Nodes representing Python files within the directory tree.
    """
    for _, _, file_nodes in directory_tree.walk():
        for node in file_nodes:
            if node.is_py_file:
                yield node


class CallResolver:
    def __init__(self):
        self.node_to_callees = {}
        self.deferred_calls = []

    def _add_function_definition(self, function_node):
        """
        Adds a function node to the call graph with no initial callees.

        This method initializes an entry in the `node_to_callees` dictionary for the
        given `function_node`, associating it with an empty set. This setup is essential
        for tracking function calls, allowing the function node to later have callees
        added as they are identified.

        Parameters
        ----------
        function_node : NodeType
            The node representing the function to be added to the call graph.
        """
        self.node_to_callees[function_node] = set()

    def _add_deferred_call(self, caller_node, callee_name):
        """
        Adds a deferred function call to the list of deferred calls.

        This method records a function call that cannot be immediately resolved, storing
        it as a tuple of the caller node and the callee name. These deferred calls are
        later resolved once all function definitions have been processed.

        Parameters
        ----------
        caller_node : NodeType
            The node representing the function that is making the call.
        callee_name : str
            The name of the function being called, which may not yet be defined or
        resolved.
        """
        call_signature = (caller_node, callee_name)
        self.deferred_calls.append(call_signature)

    def _resolve_deferred_calls(self):
        """
        Resolves deferred function calls by matching them to their corresponding
        function definitions.

        This method processes all deferred function calls that were recorded during the
        parsing of the directory tree. It attempts to match each deferred call to a
        function definition in the global scope, handling nested and recursive function
        calls. If multiple matches are found for a deferred call, or if the caller node
        is not found in the call graph, a RuntimeError is raised.

        Raises
        ------
        RuntimeError
            If more than one matching function definition is found for a deferred call,
        or if the caller node is not found in the call graph.
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

    def resolve_calls(self, directory_tree):
        """
        Resolves function calls within a directory tree by parsing Python files and
        updating the call graph.

        This method iterates over all Python file nodes in the given directory tree,
        using a `FileVisitor` to parse each file's abstract syntax tree (AST) and
        collect function definitions and deferred calls. It adds these function
        definitions to the call graph and records deferred calls for later resolution.
        After processing all files, it resolves the deferred calls by matching them to
        their corresponding function definitions.

        Parameters
        ----------
        directory_tree : DirectoryTree
            The directory tree containing Python files to be parsed and analyzed for
        function calls.
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


def _wrap(paragraph, width, indent_str):
    """
    Wraps a given paragraph to a specified width with a specified indentation.

    Parameters
    ----------
    paragraph : str
        The paragraph of text to be wrapped.
    width : int
        The maximum width of each line in the wrapped text.
    indent_str : str
        The string used to indent each line of the wrapped text.

    Returns
    -------
    str
        The wrapped text with the specified width and indentation applied.
    """
    return textwrap.fill(
        paragraph, width=width, initial_indent=indent_str, subsequent_indent=indent_str
    )


def _ensure_triple_quotes(docstring):
    """
    Ensures that a given docstring is enclosed in triple quotes.

    This function checks if the provided docstring starts and ends with triple quotes,
    either single or double. If not, it modifies the docstring to ensure it is properly
    enclosed with triple double quotes. Additionally, it ensures that the docstring
    starts and ends with a newline character if it is not already present.

    Parameters
    ----------
    docstring : str
        The docstring to be checked and modified.

    Returns
    -------
    str
        The modified docstring that is properly enclosed in triple quotes.
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
        name,
        lineno,
        col_offset,
        definition=None,
        docstring=None,
        class_node=None,
        file_node=None,
    ):
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
    def from_function_node(cls, function_node):
        """
        Creates a `CallGraphNode` instance from a given `FunctionNode`.

        This class method initializes a `CallGraphNode` using the attributes of a
        provided `FunctionNode` object, effectively converting it into a
        `CallGraphNode`.

        Parameters
        ----------
        cls : type
            The class type to instantiate, typically `CallGraphNode`.
        function_node : FunctionNode
            The function node from which to create the `CallGraphNode`.

        Returns
        -------
        CallGraphNode
            A new instance of `CallGraphNode` initialized with the attributes of the
        given `FunctionNode`.
        """
        return cls(
            name=function_node.name,
            lineno=function_node.lineno,
            col_offset=function_node.col_offset,
            definition=function_node.definition,
            docstring=function_node.docstring,
            class_node=function_node.class_node,
            file_node=function_node.file_node,
        )

    def add_child(self, child_node):
        """
        Adds a child node to the current call graph node and updates the child's parent
        list.

        This method establishes a bidirectional relationship between the current node
        and the specified child node by appending the child node to the current node's
        children list and the current node to the child's parents list.

        Parameters
        ----------
        child_node : CallGraphNode
            The node to be added as a child to the current node.
        """
        self.children.append(child_node)
        child_node.parents.append(self)

    def add_function_summary(self, summary):
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

    def add_new_docstring(self, new_docstring):
        """
        Assigns a new docstring to the `CallGraphNode` instance after validating its
        type and formatting.

        This method sets the `new_docstring` attribute of the `CallGraphNode` to the
        provided string, ensuring it is a valid string type. It then formats the
        docstring to be enclosed in triple quotes using the `_ensure_triple_quotes`
        function.

        Parameters
        ----------
        new_docstring : str
            The new docstring to be assigned to the node.

        Raises
        ------
        RuntimeError
            If the provided `new_docstring` is not of type `str`.
        """
        if not isinstance(new_docstring, str):
            raise RuntimeError("New docstring must be of type `str`")
        self.new_docstring = _ensure_triple_quotes(textwrap.dedent(new_docstring))

    def insert_new_docstring(self, max_width, indent_size):
        """
        Inserts a new docstring into the function, class, or file associated with this
        `CallGraphNode`.

        This method iterates over the possible insertion targets ('function', 'class',
        'file') and calls the `_insert_new_docstring` method to perform the insertion.
        If the target is 'class' and the `class_node` attribute is not set, it skips the
        insertion for that target.

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

    def _insert_new_docstring(self, max_width, indent_size, into):
        """
        Inserts a new docstring into the specified code section of a function, class, or
        file.

        This method updates the code of a function, class, or file by inserting a new
        docstring. It first checks if a new docstring has been provided and validates
        the target section ('function', 'class', or 'file'). The method then formats the
        new docstring according to the specified maximum width and indentation size, and
        inserts it into the appropriate location in the code.

        Parameters
        ----------
        max_width : int
            The maximum width for each line of the new docstring.
        indent_size : int
            The number of spaces to use for indentation of the new docstring.
        into : {'function', 'class', 'file'}
            The target section where the new docstring should be inserted.

        Raises
        ------
        RuntimeError
            If no new docstring has been provided or if the function is not found in the
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
            rf'(def\s+{self.name}\s*\([^)]*\)\s*:)'
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
                f"Function '{self.name}' not found in the provided code."
            )

        if into == "function":
            self.definition = new_code
        elif into == "class":
            self.class_node.definition = new_code
        elif into == "file":
            self.file_node.content = new_code

    def traverse_down(self, max_depth=None, return_start_node=True):
        """
        Traverses the call graph node hierarchy downward, collecting nodes up to a
        specified depth.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth to traverse. If None, traverses all descendants.
        return_start_node : bool, default=True
            Whether to include the starting node in the result.

        Returns
        -------
        list of tuple
            A list of tuples, each containing the depth and the node, representing the
        nodes visited during the traversal.
            If `return_start_node` is False, the starting node is excluded from the
        result.
        """
        visited = set()
        result = []

        def _traverse_down(node, current_depth):
            """
            Recursively traverses a node and its children, recording each node and its
            depth.

            Parameters
            ----------
            node : Node
                The current node being traversed.
            current_depth : int
                The depth of the current node in the traversal.

            Notes
            -----
            This function is a helper function used to perform a depth-first traversal
            of a node's children. It records each node along with its depth in a global
            result list and ensures that nodes are not revisited by maintaining a global
            set of visited nodes. The traversal can be limited by a maximum depth if
            specified.
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

    def traverse_up(self, max_depth=None, return_start_node=True):
        """
        Traverses the call graph node hierarchy upward, collecting nodes up to a
        specified depth.

        Parameters
        ----------
        max_depth : int, optional
            The maximum depth to traverse. If None, traverses all ancestors.
        return_start_node : bool, default=True
            Whether to include the starting node in the result.

        Returns
        -------
        list of tuple
            A list of tuples, each containing the depth and the node, representing the
        nodes visited during the traversal. If `return_start_node` is False, the
        starting node is excluded from the result.
        """
        visited = set()
        result = []

        def _traverse_up(node, current_depth):
            """
            Recursively traverses up the hierarchy from a given node, collecting nodes
            and their depths.

            Parameters
            ----------
            node : Node
                The current node from which to start the upward traversal.
            current_depth : int
                The current depth level of the traversal, starting from the initial
            node.

            Returns
            -------
            None
                This function does not return a value but modifies the `visited` set and
            `result` list in place.

            Notes
            -----
            This function is a helper function used within the `traverse_up` method to
            perform a depth-first search up the node hierarchy. It stops traversing when
            a node has already been visited or when the maximum depth is reached, if
            specified. The function appends each visited node along with its depth to
            the `result` list."""
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
    def __init__(self):
        self.nodes = {}

    def __repr__(self):
        return f"CallGraph({len(self.nodes)} nodes)"

    def _add_node(self, function_node):
        """
        Add a function node to the call graph if it does not already exist.

        This method checks if a function node, identified by its name, is already
        present in the call graph's nodes dictionary. If the node is not present, it
        creates a new `CallGraphNode` from the given `FunctionNode` and adds it to the
        graph. The method ensures that each function is represented only once in the
        call graph.

        Parameters
        ----------
        function_node : FunctionNode
            The function node to be added to the call graph.

        Returns
        -------
        CallGraphNode
            The `CallGraphNode` corresponding to the given `FunctionNode`, either newly
        created or already existing in the graph.
        """
        name = function_node.name
        if name not in self.nodes:
            self.nodes[name] = CallGraphNode.from_function_node(function_node)
        return self.nodes[name]

    def _add_edge(self, parent_fn_node, child_fn_node):
        """
        Establishes a parent-child relationship between two function nodes in the call
        graph.

        This method adds an edge between two nodes in the call graph, representing a
        call from the parent function node to the child function node. It ensures that
        both nodes are present in the graph by adding them if they do not already exist,
        and then links the child node to the parent node.

        Parameters
        ----------
        parent_fn_node : FunctionNode
            The function node representing the parent function in the call relationship.
        child_fn_node : FunctionNode
            The function node representing the child function in the call relationship.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        parent_node = self._add_node(parent_fn_node)
        child_node = self._add_node(child_fn_node)
        parent_node.add_child(child_node)

    def parse_graph(self, directory_tree):
        """
        Parses a directory tree to build a call graph of function relationships.

        This method initializes a `CallResolver` to process the given directory tree,
        identifying function calls and their relationships. It iterates over the
        resolved function call mappings, adding each function node to the call graph and
        establishing edges between parent and child function nodes. The method ensures
        that even solitary functions, which do not call other user-defined functions,
        are recorded in the graph. After constructing the graph, it calculates the depth
        of each node to determine the longest path from entry points.

        Parameters
        ----------
        directory_tree : DirectoryTree
            The directory tree containing Python files to be parsed and analyzed for
        function calls.

        Returns
        -------
        None

        Raises
        ------
        None
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

    def get_node(self, name):
        """
        Retrieve a node from the call graph by its name.

        This method looks up a node in the call graph's internal dictionary using the
        provided name. If the node is found, it is returned; otherwise, a KeyError is
        raised to indicate that the node does not exist in the graph.

        Parameters
        ----------
        name : str
            The name of the node to retrieve from the call graph.

        Returns
        -------
        CallGraphNode
            The node associated with the given name.

        Raises
        ------
        KeyError
            If the node with the specified name does not exist in the call graph.
        """
        node = self.nodes.get(name)
        if node is None:
            raise KeyError(f"CallGraph doesn't contain CallGraphNode '{name}'")
        return node

    def get_entrypoints(self):
        """
        Retrieve all entry point nodes in the call graph.

        An entry point node is defined as a node with no parent nodes, indicating that
        it is not called by any other node in the graph. This method iterates over all
        nodes in the call graph and selects those that have an empty list of parents.

        Returns
        -------
        list of CallGraphNode
            A list of nodes that are entry points in the call graph, meaning they have
        no parent nodes.
        """
        return [node for node in self.nodes.values() if not node.parents]

    def get_endpoints(self):
        """
        Retrieve all endpoint nodes in the call graph.

        An endpoint node is defined as a node with no child nodes, indicating that it
        does not call any other node in the graph. This method iterates over all nodes
        in the call graph and selects those that have an empty list of children.

        Returns
        -------
        list of CallGraphNode
            A list of nodes that are endpoints in the call graph, meaning they have no
        child nodes.
        """
        return [node for node in self.nodes.values() if not node.children]

    def _calculate_depths(self):
        """
        Calculate the depth of each node in the call graph.

        This method determines the depth of each node in a directed call graph, where
        nodes can have multiple parents and the graph may contain cycles due to
        recursive function calls. The depth of a node is defined as the length of the
        longest path from any entry point node (a node with no parents) to that node.
        Cycles are managed such that they do not affect the depth calculation.

        The method begins by initializing all nodes' depths to `None` to ensure a fresh
        calculation. It then computes the depth for each node starting from the entry
        points and propagates the depth values through the graph recursively. The depth
        of a node is updated only if the new calculated depth is greater than the
        current depth, ensuring the longest path is considered.

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
        # initialize with None to guarantee fresh start
        for node in self.nodes.values():
            node.depth = None

        def update_depth(node, depth, visited):
            """
            Recursively updates the depth of a node and its children in a graph.

            This function is part of a depth calculation process for nodes in a directed
            graph. It updates the depth of a given node and propagates this depth to its
            children, ensuring that each node's depth reflects the longest path from any
            entry point node to that node. The function uses a depth-first search
            approach and handles cycles by maintaining a set of visited nodes to prevent
            revisiting and artificially inflating node depths.

            Parameters
            ----------
            node : Node
                The current node whose depth is being updated.
            depth : int
                The current depth value to assign to the node.
            visited : set
                A set of nodes that have already been visited in the current depth-first
            search to avoid cycles.

            Returns
            -------
            None
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
