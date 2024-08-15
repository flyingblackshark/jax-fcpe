import flax
import torch

def convert_torch_weights(path = "fcpe_c_v001.pt"):
    state_dict = torch.load(path,map_location=torch.device('cpu'))["model"]
    params = {}
    params["input_stack_conv1.kernel"] = state_dict["input_stack.0.weight"].transpose(0,2)
    params["input_stack_conv1.bias"] = state_dict["input_stack.0.bias"]
    params["input_stack_norms.scale"] = state_dict["input_stack.1.weight"]
    params["input_stack_norms.bias"] = state_dict["input_stack.1.bias"]
    params["input_stack_conv2.kernel"] = state_dict["input_stack.3.weight"].transpose(0,2)
    params["input_stack_conv2.bias"] = state_dict["input_stack.3.bias"]
    for i in range(6):
        params[f"net.encoder_layers_{i}.conformer.net.layers_0.scale"] = state_dict[f"net.encoder_layers.{i}.conformer.net.0.weight"]
        params[f"net.encoder_layers_{i}.conformer.net.layers_0.bias"] = state_dict[f"net.encoder_layers.{i}.conformer.net.0.bias"]
        params[f"net.encoder_layers_{i}.conformer.net.layers_1.kernel"] = state_dict[f"net.encoder_layers.{i}.conformer.net.2.weight"].transpose(0,2)
        params[f"net.encoder_layers_{i}.conformer.net.layers_1.bias"] = state_dict[f"net.encoder_layers.{i}.conformer.net.2.bias"]
        params[f"net.encoder_layers_{i}.conformer.net.layers_3.kernel"] = state_dict[f"net.encoder_layers.{i}.conformer.net.4.conv.weight"].transpose(0,2)
        params[f"net.encoder_layers_{i}.conformer.net.layers_3.bias"] = state_dict[f"net.encoder_layers.{i}.conformer.net.4.conv.bias"]
        params[f"net.encoder_layers_{i}.conformer.net.layers_5.kernel"] = state_dict[f"net.encoder_layers.{i}.conformer.net.6.weight"].transpose(0,2)
        params[f"net.encoder_layers_{i}.conformer.net.layers_5.bias"] = state_dict[f"net.encoder_layers.{i}.conformer.net.6.bias"]
    params["norm.scale"] = state_dict["norm.weight"]
    params["norm.bias"] = state_dict["norm.bias"]
    params["output_proj.weight_v"] = state_dict["output_proj.weight_v"]
    params["output_proj.weight_g"] = state_dict["output_proj.weight_g"]
    params["output_proj.bias"] = state_dict["output_proj.bias"]
    params = {k: v.cpu().numpy() for k, v in params.items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    return params