#ifndef FILESAVER_
#define FILESAVER_
#include <filesystem>
#include <ATen/core/TensorBody.h>
#include <string>
#include <vtkImageData.h>
#include <torch/torch.h>
namespace torch_explorer
{

	class FileSaver
	{

	public:
		FileSaver() = delete;

		explicit FileSaver(const std::filesystem::path& directory);

		bool saveAsPNG(const torch::Tensor& tensor, const std::filesystem::path& subDirs, const std::string& filename) const;
		bool saveAsNRRD(const torch::Tensor& tensor, const std::filesystem::path& subDirs, const std::string& filename) const;
		void saveAsMHA(vtkSmartPointer<vtkImageData> image, const std::filesystem::path& subDirs, const std::string& filename) const;

	protected:

		std::filesystem::path path_;
	};
}
#endif