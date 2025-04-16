#ifndef RESIDUAL_BLOCK_H
#define RESIDUAL_BLOCK_H

#include <torch/torch.h>

namespace torch_explorer
{
	class ResidualBlock : public torch::nn::Module
	{
	public:
		ResidualBlock(int64_t in_channels, int64_t out_channels, int64_t stride = 1)
			: conv1(register_module("conv1",
			                        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
			                                          .stride(stride)
			                                          .padding(1)))),
			  conv2(register_module("conv2",
			                        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
			                                          .stride(1)
			                                          .padding(1)))),
			  bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channels))),
			  bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channels))),
			  dropout(register_module("dropout", torch::nn::Dropout(0.3)))
		{
			// Create shortcut connection if dimensions change
			if (stride != 1 || in_channels != out_channels)
			{
				shortcut = register_module("shortcut",
				                           torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
					                           .stride(stride)));
				shortcut_bn = register_module("shortcut_bn",
				                              torch::nn::BatchNorm2d(out_channels));
			}
		}

		torch::Tensor forward(torch::Tensor x)
		{
			// Ensure all operations happen on the same device as input
			auto device = x.device();
			auto identity = x;

			x = conv1->forward(x);
			x = bn1->forward(x);
			x = relu(x);
			x = dropout(x);

			x = conv2->forward(x);
			x = bn2->forward(x);

			if (shortcut && shortcut_bn)
			{
				identity = shortcut_bn->forward(shortcut->forward(identity));
			}

			x += identity;
			return relu(x);
		}

		// Optional: Add helper method to check device
		torch::Device device() const
		{
			return conv1->weight.device();
		}

	private:
		torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
		torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
		torch::nn::Conv2d shortcut{nullptr};
		torch::nn::BatchNorm2d shortcut_bn{nullptr};
		torch::nn::Dropout dropout{nullptr};
	};
}

#endif
