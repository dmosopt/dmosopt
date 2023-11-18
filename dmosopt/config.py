import sys
import importlib

def import_object_by_path(path):
    module_path, _, obj_name = path.rpartition('.')
    if module_path == '__main__' or module_path == '':
        module = sys.modules['__main__']
    else:
        module = importlib.import_module(module_path)
    return getattr(module, obj_name)