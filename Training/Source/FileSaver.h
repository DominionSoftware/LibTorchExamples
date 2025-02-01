#ifndef FILESAVER_
#define FILESAVER_
#include <filesystem>
#include <ATen/core/TensorBody.h>


class FileSaver
{

public:
	FileSaver() = delete;

	explicit FileSaver(const std::filesystem::path& directory);

	bool saveAsPNG(const at::Tensor& tensor, const std::string& filename);

protected:

	std::filesystem::path path_;
};

#endif