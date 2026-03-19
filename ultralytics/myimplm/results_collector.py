import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = "ResultNode", "ResultsCollector", "collect_results"

class ResultNode:
    """
    A node in the results tree that can hold a tensor and children nodes.
    Supports dot notation for child access and __call__ for tensor access.
    """
    def __init__(self, name="", output=None):
        self._name = name
        self._output = output
        self._children = OrderedDict()

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        raise AttributeError(f"'{self._name}' has no child named '{name}'")

    def __call__(self):
        """Returns the tensor output for this node."""
        return self._output

    def __repr__(self, indent=0):
        shape = self._output.shape if isinstance(self._output, torch.Tensor) else 'None'
        res = "  " * indent + f"{self._name}: {shape}\n"
        for child in self._children.values():
            res += child.__repr__(indent + 1)
        return res

    def __getitem__(self, key):
        """Allows access via index (e.g., res[0] or res['Layer0']) or string key."""
        if isinstance(key, int):
            skey = str(key)
            if skey in self._children:
                return self._children[skey]
            key = f"Layer{key}"

        if key in self._children:
            return self._children[key]
        raise KeyError(f"No child named {key}")

    def to_dict(self, prefix=""):
        """Converts the tree of results to a flat dictionary."""
        flat = {}
        if self._output is not None:
            flat[prefix] = self._output
        for name, child in self._children.items():
            new_prefix = f"{prefix}.{name}" if prefix else name
            flat.update(child.to_dict(new_prefix))
        return flat

class ResultsCollector:
    """
    Collects intermediate results from a PyTorch model during inference.
    Handles shared modules (like Concat or activations) correctly.
    """
    def __init__(self):
        self.hooks = []
        self.results = None
        self._hooked_modules = []

    def _multi_hook(self, m, input, output):
        """Hook that handles multiple calls to the same module instance."""
        if hasattr(m, '_rc_call_count') and m._rc_call_count < len(m._rc_node_list):
            # Clone output to avoid inplace modification issues
            if isinstance(output, torch.Tensor):
                captured_output = output.clone()
            elif isinstance(output, (list, tuple)):
                captured_output = type(output)(t.clone() if isinstance(t, torch.Tensor) else t for t in output)
            else:
                captured_output = output

            # Update the specific node in the tree that corresponds to this call count
            m._rc_node_list[m._rc_call_count]._output = captured_output
            m._rc_call_count += 1

    def _register_recursive(self, module, node):
        """Recursively registers hooks and builds the ResultNode tree."""
        # 1. Ensure the module has a hook and tracking list
        if not hasattr(module, '_rc_node_list'):
            module._rc_node_list = []
            module._rc_call_count = 0
            self.hooks.append(module.register_forward_hook(self._multi_hook))
            self._hooked_modules.append(module)

        # 2. Add THIS specific node to the list of nodes this module should populate
        module._rc_node_list.append(node)

        # 3. Recursively handle children
        # IMPORTANT: Even if this MODULE is shared, its children might be called 
        # as part of this module's forward. So we MUST build the sub-tree for this node.
        for child_name, child in module.named_children():
            child_node = ResultNode(name=child_name)
            node._children[child_name] = child_node
            self._register_recursive(child, child_node)

    def __call__(self, model, input_tensor):
        """
        Performs inference and collects results.
        """
        self.hooks = []
        self._hooked_modules = []
        self.results = ResultNode(name="root")

        target_model = model.model if hasattr(model, 'model') else model

        # Handle Ultralytics model structure
        if hasattr(target_model, 'model') and isinstance(target_model.model, (nn.Sequential, nn.ModuleList)):
            for i, layer in enumerate(target_model.model):
                layer_name = f"Layer{i}"
                layer_node = ResultNode(name=layer_name)
                self.results._children[layer_name] = layer_node
                self._register_recursive(layer, layer_node)
        else:
            self._register_recursive(target_model, self.results)

        # Initialize call counts
        for m in self._hooked_modules:
            m._rc_call_count = 0

        with torch.no_grad():
            model(input_tensor)

        # Cleanup
        for hook in self.hooks:
            hook.remove()
        for m in self._hooked_modules:
            del m._rc_node_list
            del m._rc_call_count

        return self.results

def collect_results(model, input_tensor):
    """
    Convenience function to collect all intermediate results from a model.
    """
    collector = ResultsCollector()
    return collector(model, input_tensor)

