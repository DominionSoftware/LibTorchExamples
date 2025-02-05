#ifndef CIFAR100FINE_MODULE_
#define CIFAR100FINE_MODULE_


#include <torch/torch.h>
#include <memory>
#include<inttypes.h>

#include "ResidualBlock.h"
#include <vector>


namespace torch_explorer
{
    class CIFAR100FineModule : public torch::nn::Module
    {
    public:
        CIFAR100FineModule(const std::vector<int64_t>& input_shape)
            : Module("CIFAR100Fine")
        {
            // Initial convolution with more filters
            conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_shape[0], 64, 3).padding(1)));
                
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

            // Residual blocks
            layer1 = register_module("layer1", std::make_shared<ResidualBlock>(64, 128, 2));
            layer2 = register_module("layer2", std::make_shared<ResidualBlock>(128, 256, 2));
            layer3 = register_module("layer3", std::make_shared<ResidualBlock>(256, 512, 2));
            layer4 = register_module("layer4", std::make_shared<ResidualBlock>(512, 512, 1));

            avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

            fc1 = register_module("fc1", torch::nn::Linear(512, 512));
            fc2 = register_module("fc2", torch::nn::Linear(512, 100));
            dropout = register_module("dropout", torch::nn::Dropout(0.5));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // Ensure all operations happen on the same device as input
            auto device = x.device();
            
            x = torch::relu(bn1(conv1(x)));
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);
            x = layer4->forward(x);
            x = avg_pool(x);
            x = x.view({x.size(0), -1});
            x = dropout(torch::relu(fc1(x)));
            return fc2(x);
        }

    private:
        torch::nn::Conv2d conv1{ nullptr };
        torch::nn::BatchNorm2d bn1{ nullptr };
        std::shared_ptr<ResidualBlock> layer1{ nullptr };
        std::shared_ptr<ResidualBlock> layer2{ nullptr };
        std::shared_ptr<ResidualBlock> layer3{ nullptr };
        std::shared_ptr<ResidualBlock> layer4{ nullptr };
        torch::nn::AdaptiveAvgPool2d avg_pool{ nullptr };
        torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
        torch::nn::Dropout dropout{ nullptr };
    };
}




#endif



