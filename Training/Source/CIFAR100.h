#ifndef CIFAR100_
#define CIFAR100_

#include <torch/torch.h>
#include <fstream>
#include <vector>

#include <cstdint>
#include <filesystem>

#include "ProgressBar.h"


class CIFAR100 : public torch::data::Dataset<CIFAR100>
{
public:
	enum Mode
	{
		kTrain,
		kTest
	};

	CIFAR100()
	{
		
	}

	void load(const std::filesystem::path& root, Mode mode,ProgressBar<int64_t>& progressBar)
	{
		std::string filename = mode == kTrain ? "train.bin" : "test.bin";

		std::filesystem::path rootPath = root;

		std::string file_path = rootPath.replace_filename(filename).string();


		// Try to load from file
		std::ifstream file(file_path, std::ios::binary);
		if (!file)
		{
			throw std::runtime_error("Failed to load CIFAR100 dataset");
		}

		// CIFAR-100 format:
		// <1 x coarse label><1 x fine label><3072 x pixel>
		constexpr int64_t sample_size = 1 + 1 + 3 * 32 * 32; // label + RGB image
		const int64_t num_samples = mode == kTrain ? 50000 : 10000;
		const int64_t num_channels = 3;
		const int64_t height = 32;
		const int64_t width = 32;
		const int64_t image_size = num_channels * height * width;

		// Prepare tensors to hold data
		images_ = torch::empty({ num_samples, num_channels, height, width }, torch::kByte);
		labels_ = torch::empty(num_samples, torch::kByte);
		coarse_labels_ = torch::empty(num_samples, torch::kByte);
		std::vector<uint8_t> buffer(1 + 1 + image_size);  // coarse label + fine label + image

		// Read all samples
 		for (int64_t i = 0; i < num_samples; i++)
		{
			if ((i % 100) == 0)
			{
				progressBar.progress(i, num_samples);
			}


			file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

			// First byte is coarse label, second byte is fine label
			coarse_labels_[i] =buffer[0];
			labels_[i] = buffer[1];

			// Copy image data
 		
			 
			images_[i] = torch::from_blob(
				buffer.data() + 2,  // Skip the 2 label bytes
				{ 3, 32, 32 },       // Shape: channels, height, width
				torch::kByte
			);
			/*
			int buffer_idx = 2; // Start after the labels
			for (int64_t c = 0; c < num_channels; ++c)
			{
				for (int64_t h = 0; h < height; ++h)
				{
					for (int64_t w = 0; w < width; ++w)
					{
						images_[i][c][h][w] = buffer[buffer_idx++];  
					}
				}
			}
			*/
			 
		}

		images_ = images_.to(torch::kFloat32).div_(255);
	}

	torch::data::Example<> get(size_t index) override
	{
		return {images_[index], labels_[index].clone()};
	}

	torch::optional<size_t> size() const override
	{
		return images_.size(0);
	}

	const torch::Tensor& images() const
	{
		return images_;
	}

	const torch::Tensor& targets() const
	{
		return labels_;
	}

	const torch::Tensor& coarse_targets() const
	{
		return coarse_labels_;
	}

private:
	torch::Tensor images_;
	torch::Tensor labels_;
	torch::Tensor coarse_labels_;
};

#endif
