#ifndef CIFAR100COARSE_MODULE_
#define CIFAR100COARSE_MODULE_
#include <torch/torch.h>
#include <memory>
#include "ResidualBlock.h"
#include <inttypes.h>
#include <vector>
#include "torch/nn/module.h"


namespace torch_explorer
{

    class CIFAR100CoarseModule : public torch::nn::Module
    {
    public:
        CIFAR100CoarseModule(const std::vector<int64_t>& input_shape)
            : Module("CIFAR100Coarse")
        {
            // Initial convolution with fewer filters
            conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_shape[0], 32, 3).padding(1)));

            bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));

            // Fewer residual blocks with smaller capacity
            layer1 = register_module("layer1", std::make_shared<ResidualBlock>(32, 64, 2));
            layer2 = register_module("layer2", std::make_shared<ResidualBlock>(64, 128, 2));

            avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

            fc = register_module("fc", torch::nn::Linear(128, 20));
            dropout = register_module("dropout", torch::nn::Dropout(0.3));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // Ensure all operations happen on the same device as input
            auto device = x.device();
            
            x = torch::relu(bn1(conv1(x)));
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = avg_pool(x);
            x = x.view({ x.size(0), -1 });
            x = dropout(x);
            return fc(x);
        }

        // Optional: Add helper method to check device
        torch::Device device() const {
            return conv1->weight.device();
        }

    private:
        torch::nn::Conv2d conv1{ nullptr };
        torch::nn::BatchNorm2d bn1{ nullptr };
        std::shared_ptr<ResidualBlock> layer1{ nullptr };
        std::shared_ptr<ResidualBlock> layer2{ nullptr };
        torch::nn::AdaptiveAvgPool2d avg_pool{ nullptr };
        torch::nn::Linear fc{ nullptr };
        torch::nn::Dropout dropout{ nullptr };
    };

}


#endif