#ifndef FILESAVER_
#define FILESAVER_
#include <filesystem>
#include <ATen/core/TensorBody.h>

namespace torch_explorer
{

	class FileSaver
	{

	public:
		FileSaver() = delete;

		explicit FileSaver(const std::filesystem::path& directory);

		bool saveAsPNG(const at::Tensor& tensor, const std::filesystem::path& subDirs, const std::string& filename);

	protected:

		std::filesystem::path path_;
	};
}
#endif