#ifndef CIFAR100_
#define CIFAR100_

#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <filesystem>


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

	explicit CIFAR100(const std::filesystem::path& root, Mode mode = kTrain)
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

		// Prepare tensors to hold data
		images_ = torch::empty({num_samples, 3, 32, 32}, torch::kByte);
		targets_ = torch::empty(num_samples, torch::kByte);
		coarse_targets_ = torch::empty(num_samples, torch::kByte);

		// Read all samples
		std::vector<uint8_t> sample(sample_size);
		for (int64_t i = 0; i < num_samples; i++)
		{
			file.read(reinterpret_cast<char*>(sample.data()), sample_size);

			// First byte is coarse label, second byte is fine label
			coarse_targets_[i] = sample[0];
			targets_[i] = sample[1];

			// Copy image data
			auto image_data = images_[i];
			int pixel_ptr = 2; // Start after the labels
			for (int64_t c = 0; c < 3; c++)
			{
				for (int64_t x = 0; x < 32; x++)
				{
					for (int64_t y = 0; y < 32; y++)
					{
						image_data[c][x][y] = sample[pixel_ptr++];
					}
				}
			}
		}

		images_ = images_.to(torch::kFloat32).div_(255);
	}

	torch::data::Example<> get(size_t index) override
	{
		return {images_[index], targets_[index].clone()};
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
		return targets_;
	}

	const torch::Tensor& coarse_targets() const
	{
		return coarse_targets_;
	}

private:
	torch::Tensor images_, targets_, coarse_targets_;
};

#endif
