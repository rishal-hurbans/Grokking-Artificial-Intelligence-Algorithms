import ast
import contextlib
import importlib.util
import os
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def temporary_cwd(path):
    original_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


@contextlib.contextmanager
def temporary_sys_path(path):
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path)


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with temporary_sys_path(path.parent):
        spec.loader.exec_module(module)
    return module


def load_module_from_source(name, relative_path, skip_node):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not skip_node(node)]
    ast.fix_missing_locations(tree)

    module = types.ModuleType(name)
    module.__file__ = str(path)
    with temporary_sys_path(path.parent):
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_definitions_from_source(name, relative_path):
    return load_module_from_source(
        name,
        relative_path,
        lambda node: not isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef),
        ),
    )
