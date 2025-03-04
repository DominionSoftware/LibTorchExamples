#ifndef COVID19_MODULE_
#define COVID19_MODULE_

#include <torch/torch.h>
#include <torchvision.h>
#include <vector>

namespace torch_explorer
{
    // ResNet18-based module for Covid19 classification
    class Covid19Module : public torch::nn::Module
    {
    public:
        Covid19Module(size_t num_classes = 4, bool pretrained = true) 
        {
            // Load a pretrained ResNet18 model
            auto resnet = torch::vision::models::resnet18(pretrained);
            
            // Freeze all parameters if using pretrained model
            if (pretrained) {
                for (auto& param : resnet->parameters()) {
                    param.requires_grad_(false);
                }
            }
            
            // Get number of features in the final layer
            int64_t num_features = resnet->fc->weight.sizes()[1];
            
            // Replace the final fully connected layer
            resnet->fc = torch::nn::Linear(num_features, num_classes);
            
            // Register the model
            backbone = resnet;
            register_module("backbone", backbone);
        }
        
        torch::Tensor forward(torch::Tensor x)
        {
            return backbone->forward(x);
        }
        

       torch::Tensor forwardHierarchical(torch::Tensor x)
        {
            return forward(x);
         }

    private:
        torch::nn::Sequential backbone{nullptr};
    };
}

#endif 