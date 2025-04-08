 
#include "FileSaver.h"


#include "png.h"
#include <torch/torch.h>
#include <vector>

using namespace torch_explorer;


FileSaver::FileSaver(const std::filesystem::path& directory) : path_(directory)
{

}

bool FileSaver::saveAsPNG(const torch::Tensor& tensor,const std::filesystem::path& subDirs,const std::string& filename)
{
	auto cpu_tensor = tensor.to(torch::kCPU);

	cpu_tensor = cpu_tensor.permute({ 1, 2, 0 }).contiguous();

	int64_t height = cpu_tensor.size(0);
	int64_t width = cpu_tensor.size(1);
	int64_t channels = cpu_tensor.size(2);

	std::filesystem::path localPath = path_ / subDirs;

	if (!std::filesystem::exists(localPath))
	{
		std::error_code ec;
		const bool ok = std::filesystem::create_directories(localPath,ec);
		if (!ok)
		{
			std::cerr << "Error creating directories: " << ec.message() << " (" << ec.value() << ")\n";
		}
	}
	localPath = localPath / "";

	localPath.replace_filename(filename);


	FILE* fp = fopen(localPath.string().c_str(), "wb");
	if (!fp) {
		return false;
	}


	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!png_ptr) {
		int e = fclose(fp);
		return false;
	}


	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr, nullptr);
		int e = fclose(fp);
		return false;
	}


	png_init_io(png_ptr, fp);

 
	png_set_IHDR(png_ptr, info_ptr, width, height, 8,
		PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png_ptr, info_ptr);


	std::vector<png_bytep> row_pointers(height);
	auto tensor_data = cpu_tensor.data_ptr<uint8_t>();

	for (int64_t y = 0; y < height; y++) {
		row_pointers[y] = (png_bytep)(tensor_data + y * width * channels);
	}

	png_write_image(png_ptr, row_pointers.data());
	png_write_end(png_ptr, nullptr);

	// Cleanup
	png_destroy_write_struct(&png_ptr, &info_ptr);
	int e = fclose(fp);

	return true;

}
 