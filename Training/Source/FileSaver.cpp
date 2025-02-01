 
#include "FileSaver.h"


#include "png.h"
#include <torch/torch.h>
#include <vector>
 

FileSaver::FileSaver(const std::filesystem::path& directory) : path_(directory)
{

}

bool FileSaver::saveAsPNG(const torch::Tensor& tensor,const std::string& filename)
{
	auto cpu_tensor = tensor.to(torch::kCPU);

	cpu_tensor = cpu_tensor.permute({ 1, 2, 0 }).contiguous();

	int height = cpu_tensor.size(0);
	int width = cpu_tensor.size(1);
	int channels = cpu_tensor.size(2);

	if (!std::filesystem::exists(path_))
	{
		bool ok = std::filesystem::create_directories(path_);
		if (!ok)
		{
			return false;
		}
	}

	path_.replace_filename(filename);


	FILE* fp = fopen(path_.string().c_str(), "wb");
	if (!fp) {
		return false;
	}


	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!png_ptr) {
		fclose(fp);
		return false;
	}


	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_write_struct(&png_ptr, nullptr);
		fclose(fp);
		return false;
	}


	png_init_io(png_ptr, fp);

 
	png_set_IHDR(png_ptr, info_ptr, width, height, 8,
		PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png_ptr, info_ptr);


	std::vector<png_bytep> row_pointers(height);
	auto tensor_data = cpu_tensor.data_ptr<uint8_t>();

	for (int y = 0; y < height; y++) {
		row_pointers[y] = (png_bytep)(tensor_data + y * width * channels);
	}

	png_write_image(png_ptr, row_pointers.data());
	png_write_end(png_ptr, nullptr);

	// Cleanup
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);

	return true;

}
 