import torch
import torch.onnx
import argparse
import os

def infer_input_shape(model):
    # Attempt to infer the input shape from the first layer of the model
    # This method might need adjustments for different model architectures
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            # Assuming the layer is a Conv2d layer, which is common as the first layer
            # The shape is inferred as: (batch_size, in_channels, height, width)
            # Here, we assume a default size (224, 224) for height and width, which might need to be adjusted
            return (1, layer.in_channels, 224, 224)
        elif isinstance(layer, torch.nn.Linear):
            # If the first layer is a Linear layer, we infer it's a non-image input model
            # The input shape is: (batch_size, in_features)
            return (1, layer.in_features)
    # If unable to infer, return a default or raise an error
    raise ValueError("Unable to infer input shape from the first layer of the model.")

def convert_pt_to_onnx(pt_file_path, onnx_file_path):
    # Load the PyTorch model
    model = torch.load(pt_file_path)
    model.eval()  # Set the model to evaluation mode
    
    # Infer the input shape from the model
    input_shape = infer_input_shape(model)
    
    # Create a dummy input tensor appropriate for the model
    dummy_input = torch.randn(input_shape)
    
    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_file_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    
    print(f"Model has been converted to ONNX format and saved to {onnx_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to ONNX format.")
    parser.add_argument("pt_file_path", type=str, help="Path to the .pt model file")
    parser.add_argument("onnx_file_path", type=str, help="Path where the ONNX model will be saved")
    
    args = parser.parse_args()

    # Ensure the .pt file exists
    if not os.path.isfile(args.pt_file_path):
        raise FileNotFoundError(f"The specified .pt file does not exist: {args.pt_file_path}")
    
    # Convert the model
    convert_pt_to_onnx(args.pt_file_path, args.onnx_file_path)
