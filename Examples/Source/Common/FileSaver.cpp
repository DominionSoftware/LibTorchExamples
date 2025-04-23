 
#include "FileSaver.h"


#include "png.h"
#include <torch/torch.h>
#include <vector>

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkNrrdImageIO.h>       
#include <vtkMetaImageWriter.h>
#include "vtkImageExport.h"
#include "itkVTKImageImport.h"
#include "itkImageFileWriter.h"
#include "itkNrrdImageIO.h"



using namespace torch_explorer;


FileSaver::FileSaver(const std::filesystem::path& directory) : path_(directory)
{

}

bool FileSaver::saveAsPNG(const torch::Tensor& tensor,const std::filesystem::path& subDirs,const std::string& filename) const
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

bool FileSaver::saveAsNRRD(const torch::Tensor& tensor, const std::filesystem::path& subDirs, const std::string& filename) const
{
    // Ensure tensor is on CPU and convert to contiguous memory layout
    auto cpu_tensor = tensor.to(torch::kCPU).contiguous();

    // Get tensor dimensions
    auto sizes = cpu_tensor.sizes().vec();

    // Check if this is a volume (expecting 5D tensor: batch, channel, depth, height, width)
    if (sizes.size() != 5) {
        std::cerr << "Expected 5D tensor (batch, channel, depth, height, width) but got "
            << sizes.size() << "D tensor" << std::endl;
        return false;
    }

    int batch = sizes[0];
    int channels = sizes[1];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];

    // For simplicity, we'll only save the first batch and channel
    if (batch > 1 || channels > 1) {
        std::cout << "Warning: Only saving first batch and channel of the tensor" << std::endl;
    }

    // Create local path
    std::filesystem::path localPath = path_ / subDirs;

    if (!std::filesystem::exists(localPath)) {
        std::error_code ec;
        const bool ok = std::filesystem::create_directories(localPath, ec);
        if (!ok) {
            std::cerr << "Error creating directories: " << ec.message() << " (" << ec.value() << ")\n";
            return false;
        }
    }

    localPath = localPath / "";
    localPath.replace_filename(filename);

    // If filename doesn't end with .nrrd, add it
    if (localPath.extension() != ".nrrd") {
        localPath += ".nrrd";
    }

    // Set up the ITK image
    using PixelType = float;
    const unsigned int Dimension = 3;
    using ImageType = itk::Image<PixelType, Dimension>;

    ImageType::Pointer image = ImageType::New();

    // Set image size
    ImageType::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = depth;

    // Set image region
    ImageType::RegionType region;
    region.SetSize(size);

    // Set image spacing (default to 1.0 if not available)
    ImageType::SpacingType spacing;
    spacing.Fill(1.0);

    // Set image origin (default to 0.0 if not available)
    ImageType::PointType origin;
    origin.Fill(0.0);

    // Set image direction (default to identity if not available)
    ImageType::DirectionType direction;
    direction.SetIdentity();

    // Apply all the settings to the image
    image->SetRegions(region);
    image->SetSpacing(spacing);
    image->SetOrigin(origin);
    image->SetDirection(direction);
    image->Allocate();

    // Access the first batch and channel of the tensor
    auto tensor_slice = cpu_tensor[0][0];
    auto tensor_data = tensor_slice.data_ptr<float>();

    // Copy data from tensor to ITK image
    itk::ImageRegionIterator<ImageType> imageIterator(image, image->GetLargestPossibleRegion());

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate index in the flat tensor array
                size_t tensorIndex = (z * height * width) + (y * width) + x;

                // Set ITK image pixel
                ImageType::IndexType pixelIndex;
                pixelIndex[0] = x;
                pixelIndex[1] = y;
                pixelIndex[2] = z;

                image->SetPixel(pixelIndex, tensor_data[tensorIndex]);
            }
        }
    }

    // Set up the writer
    using WriterType = itk::ImageFileWriter<ImageType>;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(localPath.string());
    writer->SetInput(image);

    try {
        writer->Update();
    }
    catch (itk::ExceptionObject& error) {
        std::cerr << "Error: " << error << std::endl;
        return false;
    }

    std::cout << "Successfully saved NRRD file to: " << localPath.string() << std::endl;
    return true;
}


void FileSaver::saveAsMHA(vtkSmartPointer<vtkImageData> image, const std::filesystem::path& subDirs, const std::string& filename) const
{
    // Construct the full path by combining path_, subDirs, and filename
    std::filesystem::path fullPath = path_ / subDirs / (filename + ".mha");

    // Create the directory structure if it doesn't exist
    std::filesystem::create_directories(fullPath.parent_path());

    // Create a writer for the MetaImage format (.mha)
    vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();

    // Set the input image data
    writer->SetInputData(image);

    // Set the filename for the output
    writer->SetFileName(fullPath.string().c_str());


    // Write the file to disk
    writer->Write();
}


void FileSaver::saveAsNRRD(vtkSmartPointer<vtkImageData> vtkImage, const std::filesystem::path& subDirs, const std::string& filename) const
{
    // Construct the full path by combining path_, subDirs, and filename
    std::filesystem::path fullPath = path_ / subDirs / (filename + ".nrrd");
    
    // Create the directory structure if it doesn't exist
    std::filesystem::create_directories(fullPath.parent_path());
    
    // Determine image properties from the VTK image
    int* dimensions = vtkImage->GetDimensions();
    int numComponents = vtkImage->GetNumberOfScalarComponents();
    int scalarType = vtkImage->GetScalarType();
    
    // Create appropriate ITK image type based on VTK image properties
    // For simplicity, we'll use a fixed type here, but you could add logic to handle different types
    using PixelType = float;  // Most common type, adjust if needed
    const unsigned int Dimension = 3;  // Standard for medical images
    using ITKImageType = itk::Image<PixelType, Dimension>;
    
    // Convert VTK to ITK image using the bridge
    ITKImageType::Pointer itkImage;
    
    // Wrap the VTK image data in an ITK-compatible wrapper
    vtkImageExport* exporter = vtkImageExport::New();
    exporter->SetInputData(vtkImage);
    exporter->Update();
    
    // Set up ITK importer to receive data from VTK exporter
    using ImporterType = itk::VTKImageImport<ITKImageType>;
    ImporterType::Pointer importer = ImporterType::New();
    
    // Connect the VTK exporter to the ITK importer
    importer->SetUpdateInformationCallback(exporter->GetUpdateInformationCallback());
    importer->SetPipelineModifiedCallback(exporter->GetPipelineModifiedCallback());
    importer->SetWholeExtentCallback(exporter->GetWholeExtentCallback());
    importer->SetSpacingCallback(exporter->GetSpacingCallback());
    importer->SetOriginCallback(exporter->GetOriginCallback());
    importer->SetScalarTypeCallback(exporter->GetScalarTypeCallback());
    importer->SetNumberOfComponentsCallback(exporter->GetNumberOfComponentsCallback());
    importer->SetPropagateUpdateExtentCallback(exporter->GetPropagateUpdateExtentCallback());
    importer->SetUpdateDataCallback(exporter->GetUpdateDataCallback());
    importer->SetDataExtentCallback(exporter->GetDataExtentCallback());
    importer->SetBufferPointerCallback(exporter->GetBufferPointerCallback());
    importer->SetCallbackUserData(exporter->GetCallbackUserData());
    
    // Now do the actual import
    importer->Update();
    itkImage = importer->GetOutput();
    
    // Create a writer for the NRRD format (.nrrd)
    using WriterType = itk::ImageFileWriter<ITKImageType>;
    WriterType::Pointer writer = WriterType::New();
    
    // Set the input image data
    writer->SetInput(itkImage);
    
    // Set the filename for the output
    writer->SetFileName(fullPath.string());
    
    // Use NRRD IO factory
    writer->SetImageIO(itk::NrrdImageIO::New());
    
    // Write the file to disk
    try
    {
        writer->Update();
    }
    catch (itk::ExceptionObject & error)
    {
        std::cerr << "Error writing NRRD file: " << error << std::endl;
        throw;
    }
    
    // Clean up
    exporter->Delete();
}