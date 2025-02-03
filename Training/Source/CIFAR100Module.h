#ifndef CIFAR100Module_
#define CIFAR100Module_

#include <torch/torch.h>
#include <memory>
#include <tuple>
namespace torch_explorer
{

	class CIFAR100Module : public torch::nn::Module
	{
	public:
		CIFAR100Module(const std::vector<int64_t>& input_shape, int64_t num_classes)
		{
			// Calculate dimensions for fully connected layer
			int64_t height = input_shape[1];
			int64_t width = input_shape[2];

			// Shared Feature Extraction Layers
			conv1 = register_module("conv1", torch::nn::Conv2d(
				torch::nn::Conv2dOptions(input_shape[0], 64, 3).padding(1)));
			bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

			height /= 2; // After pooling
			width /= 2;

			conv2 = register_module("conv2", torch::nn::Conv2d(
				torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
			bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));

			height /= 2; // After pooling
			width /= 2;

			conv3 = register_module("conv3", torch::nn::Conv2d(
				torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
			bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));

			height /= 2; // After pooling
			width /= 2;

			// Calculate flattened size
			int64_t fc_input_size = 256 * height * width;

			// Shared fully connected layers
			shared_fc = register_module("shared_fc", torch::nn::Linear(fc_input_size, 1024));
			dropout = register_module("dropout", torch::nn::Dropout(0.5));

			// Coarse classification branch (20 superclasses)
			coarse_fc1 = register_module("coarse_fc1", torch::nn::Linear(1024, 512));
			coarse_fc2 = register_module("coarse_fc2", torch::nn::Linear(512, 20));

			// Fine classification branch (100 classes)
			fine_fc1 = register_module("fine_fc1", torch::nn::Linear(1024, 512));
			fine_fc2 = register_module("fine_fc2", torch::nn::Linear(512, 100));
		}

		std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
		{
			// Shared feature extraction
			x = relu(bn1(conv1(x)));
			x = max_pool2d(x, 2);

			x = relu(bn2(conv2(x)));
			x = max_pool2d(x, 2);

			x = relu(bn3(conv3(x)));
			x = max_pool2d(x, 2);

			x = x.flatten(1);

			// Shared fully connected processing
			x = relu(shared_fc(x));
			x = dropout(x);

			// Coarse classification branch
			auto coarse_features = relu(coarse_fc1(x));
			auto coarse_out = coarse_fc2(coarse_features);

			// Fine classification branch
			auto fine_features = relu(fine_fc1(x));
			auto fine_out = fine_fc2(fine_features);

			return std::make_tuple(coarse_out, fine_out);
		}

	private:
		// Shared layers
		torch::nn::Conv2d conv1{ nullptr };
		torch::nn::Conv2d conv2{ nullptr };
		torch::nn::Conv2d conv3{ nullptr };
		torch::nn::BatchNorm2d bn1{ nullptr };
		torch::nn::BatchNorm2d bn2{ nullptr };
		torch::nn::BatchNorm2d bn3{ nullptr };
		torch::nn::Linear shared_fc{ nullptr };
		torch::nn::Dropout dropout{ nullptr };

		// Coarse classification layers
		torch::nn::Linear coarse_fc1{ nullptr };
		torch::nn::Linear coarse_fc2{ nullptr };

		// Fine classification layers
		torch::nn::Linear fine_fc1{ nullptr };
		torch::nn::Linear fine_fc2{ nullptr };
	};
}
#endif
