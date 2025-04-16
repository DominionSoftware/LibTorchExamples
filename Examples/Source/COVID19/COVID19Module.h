#ifndef COVID19_MODULE_
#define COVID19_MODULE_

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

namespace torch_explorer
{
    // Standard torch::nn::Module-based Covid19 classification module
    class Covid19Module : public torch::nn::Module
    {
    public:
        const size_t num_classes = 4;

        Covid19Module(const std::string& model_path ) :
            device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {
            // Determine device (CPU/CUDA)
 
            try {
                // Load pretrained features
                pretrained_model = torch::jit::load(model_path);
                pretrained_model.to(device);

                // Create and register the classifier
                classifier = register_module("classifier", torch::nn::Linear(512, num_classes));
                classifier->to(device);

                std::cout << "Covid19Module initialized successfully with " << num_classes << " classes" << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading the model from " << model_path << ": " << e.what() << std::endl;
                throw; // Rethrow the exception
            }
        }

        void to(torch::Device device_arg, bool non_blocking = false) override
        {
            device = device_arg;

            // Call the parent class to() method to move parameters and buffers
            torch::nn::Module::to(device, non_blocking);

            // Also move the TorchScript module
            pretrained_model.to(device);

            std::cout << "Model moved to device: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // Ensure input is on the correct device
            x = x.to(device);

            // Prepare input for TorchScript model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x);

            // Get features from the pretrained model
            torch::Tensor features;
            try {
                features = pretrained_model.forward(inputs).toTensor();
            }
            catch (const c10::Error& e) {
                std::cerr << "Error during forward pass: " << e.what() << std::endl;
                // Create a dummy tensor with the right dimensions for debugging
                return torch::zeros({ x.size(0), classifier->options.out_features() }).to(device);
            }

            // Apply classifier
            return classifier(features);
        }

        torch::Tensor forwardHierarchical(torch::Tensor x)
        {
            return forward(x);
        }

    private:
        torch::jit::Module pretrained_model;
        torch::nn::Linear classifier{ nullptr };
        torch::Device device;
    };
}

#endif