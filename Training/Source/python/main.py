# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import monai
import os
import json


def printmodel(path):
    model = torch.load(
        "D:/Projects/LibTorchExamples/Training/Source/PreTrainedModels/wholeBody_ct_segmentation/models/model.pt",
        map_location=torch.device('cpu'))
    #print(model)


try:
    with open(
            "D:/Projects/LibTorchExamples/Training/Source/PreTrainedModels/wholeBody_ct_segmentation/configs/metadata.json",
            'r') as f:
        metadata = json.load(f)
    print(json.dumps(metadata, indent=2))
except:
    print("Could not load metadata.json")


def convert(path):
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "configs", "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Model info from metadata: {metadata.get('model_info', {})}")


    state_dict = torch.load(path,map_location=torch.device('cpu'))
    print(f"Loaded object type: {type(state_dict)}")
    if hasattr(state_dict, 'eval'):
        model = state_dict
    else:
        from monai.networks.nets import SegResNet

        out_channels = metadata.get("model_info", {}).get("out_channels", 105)
         #num_classes = metadata.get("model_info", {}).get("num_classes", 105)

        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=32,
            in_channels=1,
            out_channels=out_channels,  # For whole body segmentation (adjust based on metadata)
            dropout_prob=0.0,
        )
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                # If direct loading fails, we might need more complex logic
                # based on the specific structure of your model
    try:
        model.eval()  # Set to evaluation mode
        example_input = torch.randn(1, 1, 96, 96, 96)
        traced_model = torch.jit.trace(model, example_input)
        dir_path = os.path.dirname(path)

        file_name_no_ext, file_ext = os.path.splitext(os.path.basename(path))

        full_path = os.path.join(dir_path, file_name_no_ext + ".ts")
        traced_model.save(full_path)

    except Exception as e:
        print(f"Error tracing model: {e}")

        # If tracing fails, try scripting instead
        try:
            scripted_model = torch.jit.script(model)

            # Save the scripted model
            dir_path = os.path.dirname(path)

            file_name_no_ext, file_ext = os.path.splitext(os.path.basename(path))

            output_path = os.path.join(dir_path, file_name_no_ext + ".ts")
            scripted_model.save(output_path)
            print(f"Successfully saved scripted model to {output_path}")


        except Exception as e2:
            print(f"Error scripting model: {e2}")
            raise e2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    convert("D:/Projects/LibTorchExamples/Training/Source/PreTrainedModels/wholeBody_ct_segmentation/models/model.pt")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
