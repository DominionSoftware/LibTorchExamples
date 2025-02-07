#ifndef CIFAR100_
#define CIFAR100_

#include <torch/torch.h>
#include <fstream>
#include <vector>

#include <cstdint>
#include <filesystem>

#include "FileSaver.h"
#include "ProgressBar.h"
#include "CIFAR100ClassNames.h"

namespace torch_explorer
{
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

		void load(const std::filesystem::path& root, Mode mode, ProgressBar<int64_t>& progressBar, std::shared_ptr<FileSaver> fileSaver = nullptr)
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
			class_labels_ = torch::empty(num_samples, torch::kByte);
			superclass_labels_ = torch::empty(num_samples, torch::kByte);
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
				superclass_labels_[i] = buffer[0];
				class_labels_[i] = buffer[1];

				// Copy image data


				images_[i] = torch::from_blob(
					buffer.data() + 2,  // Skip the 2 label bytes
					{ 3, 32, 32 },       // Shape: channels, height, width
					torch::kByte
				);

				if (fileSaver != nullptr)
				{
					std::string s;
					std::stringstream ss(s);
					int superClass = superclass_labels_[i].item<int>();
					std::cout << superClass << std::endl;

					std::string superclassName = CIFAR100ClassNames::instance().getSuperclassName(superClass);
					int fineClass = class_labels_[i].item<int>();
					std::cout << fineClass << std::endl;
					std::string fineclassName = CIFAR100ClassNames::instance().getFineClassName(fineClass);

					ss << i << "_" << superclassName << "_" << fineclassName << "_image.png";
					std::filesystem::path subdirs(superclassName);
					subdirs = subdirs / fineclassName;


					fileSaver->saveAsPNG(images_[i], subdirs, ss.str());
				}
			}
			progressBar.progress(num_samples, num_samples);

			images_ = images_.to(torch::kFloat32).div_(255);
		}

		torch::data::Example<> get(size_t index) override
		{
			return { images_[index], class_labels_[index].clone() };
		}

		torch::optional<size_t> size() const override
		{
			return images_.size(0);
		}

		const torch::Tensor& images() const
		{
			return images_;
		}

		const torch::Tensor& classLabels() const
		{
			return class_labels_;
		}

		const torch::Tensor& superclassLabels() const
		{
			return superclass_labels_;
		}

	private:
		torch::Tensor images_;
		torch::Tensor class_labels_;
		torch::Tensor superclass_labels_;
	};
}
#endif
