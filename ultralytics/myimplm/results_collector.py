from collections import OrderedDict

import torch
from torch import nn

__all__ = "ResultNode", "ResultsCollector", "collect_results"


class ResultNode:
    """A node in the results tree that can hold a tensor and children nodes. Supports dot notation for child access and
    __call__ for tensor access.
    """

    def __init__(self, name="", output=None):
        self._name = name
        self._output = output
        self._input = None  # Added to hold input
        self._children = OrderedDict()

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        raise AttributeError(f"'{self._name}' has no child named '{name}'")

    def __call__(self):
        """Returns the tensor output for this node."""
        return self._output

    def input(self):
        """Returns the tensor input for this node."""
        return self._input

    def __repr__(self, indent=0):
        shape = self._output.shape if isinstance(self._output, torch.Tensor) else "None"
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
    """Collects intermediate results from a PyTorch model during inference. Handles shared modules (like Concat or
    activations) correctly.
    """

    def __init__(self):
        self.hooks = []
        self.results = None
        self._hooked_modules = []

    def _multi_hook(self, m, input, output):
        """Hook that handles multiple calls to the same module instance."""
        if hasattr(m, "_rc_call_count") and m._rc_call_count < len(m._rc_node_list):
            # 1. Capture Input
            if isinstance(input, (list, tuple)) and len(input) > 0:
                in_data = input[0]  # Usually first arg
            else:
                in_data = input

            if isinstance(in_data, torch.Tensor):
                captured_input = in_data.clone()
            else:
                captured_input = in_data

            # 2. Capture Output
            if isinstance(output, torch.Tensor):
                captured_output = output.clone()
            elif isinstance(output, (list, tuple)):
                captured_output = type(output)(t.clone() if isinstance(t, torch.Tensor) else t for t in output)
            else:
                captured_output = output

            # Update the specific node
            node = m._rc_node_list[m._rc_call_count]
            node._output = captured_output
            node._input = captured_input
            m._rc_call_count += 1

    def _register_recursive(self, module, node):
        """Recursively registers hooks and builds the ResultNode tree."""
        # 1. Ensure the module has a tracking list
        if not hasattr(module, "_rc_node_list"):
            module._rc_node_list = []
            # We only register the hook ONCE per module instance
            self.hooks.append(module.register_forward_hook(self._multi_hook))
            self._hooked_modules.append(module)

        # 2. Add THIS specific node to the list of nodes this module should populate.
        # If a module is shared, it will have multiple nodes in its list.
        module._rc_node_list.append(node)

        # 3. Recursively handle children
        for child_name, child in module.named_children():
            child_node = ResultNode(name=child_name)
            node._children[child_name] = child_node
            self._register_recursive(child, child_node)

    def __call__(self, model, input_tensor):
        """Performs inference and collects results."""
        self.hooks = []
        self._hooked_modules = []
        self.results = ResultNode(name="root")

        target_model = model.model if hasattr(model, "model") else model

        # Handle Ultralytics model structure
        if hasattr(target_model, "model") and isinstance(target_model.model, (nn.Sequential, nn.ModuleList)):
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
    """Convenience function to collect all intermediate results from a model."""
    collector = ResultsCollector()
    return collector(model, input_tensor)
