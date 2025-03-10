#include "MONAIDataLoader.h"

#include <regex>
#include <torch/torch.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>


using namespace torch_explorer;


MonaiDataLoader::MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode) : bundlePath_(bundlePath),
    trainMode_(trainMode)
{
    std::vector<std::filesystem::path> configPaths =
    {
        bundlePath / "configs" / "dataset_0.json",
        bundlePath / "configs" / "dataset.json",
        bundlePath / "dataset.json"
    };

    for (const auto& path : configPaths)
    {
        if (exists(path))
        {
            std::ifstream configFile(path);
            configFile >> config_;
            std::cout << "Loaded dataset configuration from: " << path.string() << std::endl;
            break;
        }
    }
}


nlohmann::json MonaiDataLoader::getMetadata() const
{
    nlohmann::json metadata;
    std::vector<std::filesystem::path> metadataPaths =
    {
        bundlePath_ / "metadata.json",
        bundlePath_ / "configs" / "metadata.json"
    };

    for (const auto& path : metadataPaths)
    {
        if (exists(path))
        {
            std::ifstream metadataFile(path);
            metadataFile >> metadata;
            break;
        }
    }

    return metadata;
}


std::vector<std::pair<std::string, std::string>> MonaiDataLoader::loadDataPairs()
{
    std::vector<std::pair<std::string, std::string>> dataPairs;


    if (!config_.empty())
    {
        dataPairs = loadFromConfig();
    }


    if (dataPairs.empty())
    {
        dataPairs = scanForImageLabelPairs();
    }

    std::cout << "Found " << dataPairs.size() << " image-label pairs" << std::endl;
    return dataPairs;
}

torch::Tensor MonaiDataLoader::loadImage(const std::string& path, bool isLabel)
{
    auto extension = std::filesystem::path(path).extension().string();

    if (extension == ".dcm")
    {
        return loadDicomImage(path, isLabel);
    }
    // TODO other formats.
    throw std::runtime_error("Unsupported file format: " + extension);
}

torch::Tensor MonaiDataLoader::loadDicomImage(const std::string& path, bool isLabel)
{
    std::vector<std::string> dicomFiles;

    if (std::filesystem::is_directory(path))
    {
        // Get all DICOM files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(path))
        {
            if (entry.is_regular_file())
            {
                auto filePath = entry.path().string();
                DcmFileFormat fileFormat;
                if (fileFormat.loadFile(filePath.c_str()).good())
                {
                    dicomFiles.push_back(filePath);
                }
            }
        }
    }
    else
    {
        // Single DICOM file
        dicomFiles.push_back(path);
    }

    if (dicomFiles.empty())
    {
        throw std::runtime_error("No valid DICOM files found in: " + path);
    }


    // TODO Sort by position not instance number;
    //


    // Sort DICOM files by instance number for correct 3D ordering
    std::sort(dicomFiles.begin(), dicomFiles.end(),
              [](const std::string& a, const std::string& b)
              {
                  DcmFileFormat fileA, fileB;
                  int instanceA = 0, instanceB = 0;

                  if (fileA.loadFile(a.c_str()).good() &&
                      fileA.getDataset()->findAndGetSint32(DCM_InstanceNumber, instanceA).good() &&
                      fileB.loadFile(b.c_str()).good() &&
                      fileB.getDataset()->findAndGetSint32(DCM_InstanceNumber, instanceB).good())
                  {
                      return instanceA < instanceB;
                  }

                  // Fallback to filename comparison
                  return a < b;
              });

    // Parameters for the 3D volume
    unsigned long width = 0, height = 0;
    size_t depth = dicomFiles.size();
    std::vector<float> pixelData;

    // Read each slice
    for (const auto& filePath : dicomFiles)
    {
        DcmFileFormat fileFormat;
        OFCondition status = fileFormat.loadFile(filePath.c_str());

        if (status.good())
        {
            DicomImage image(&fileFormat, EXS_Unknown);

            if (image.getStatus() == EIS_Normal)
            {
                // If first slice, get dimensions
                if (width == 0)
                {
                    width = image.getWidth();
                    height = image.getHeight();
                    pixelData.reserve(width * height * depth);
                }

                // Access raw pixel data
                const DiPixel* pixelMatrix = image.getInterData();
                if (pixelMatrix)
                {
                    auto data = const_cast<void*>(pixelMatrix->getData());
                    if (data)
                    {
                        // Get technical parameters
                        int bitsAllocated = 0;
                        fileFormat.getDataset()->findAndGetSint32(DCM_BitsAllocated, bitsAllocated);

                        if (!isLabel)
                        {
                            // For image data, convert to Hounsfield Units
                            double rescaleIntercept = 0.0;
                            double rescaleSlope = 1.0;
                            fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, rescaleIntercept);
                            fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleSlope, rescaleSlope);

                            // Process pixel data based on bit depth
                            if (bitsAllocated == 16)
                            {
                                // Get pixel representation (signed or unsigned)
                                int pixelRepresentation = 0;
                                fileFormat.getDataset()->findAndGetSint32(DCM_PixelRepresentation, pixelRepresentation);

                                if (pixelRepresentation == 0)
                                {
                                    // Unsigned data
                                    auto pixelValues = static_cast<const uint16_t*>(data);
                                    for (unsigned long i = 0; i < width * height; i++)
                                    {
                                        float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope +
                                            rescaleIntercept;
                                        pixelData.push_back(hounsfield);
                                    }
                                }
                                else
                                {
                                    // Signed data
                                    auto pixelValues = static_cast<const int16_t*>(data);
                                    for (unsigned long i = 0; i < width * height; i++)
                                    {
                                        float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope +
                                            rescaleIntercept;
                                        pixelData.push_back(hounsfield);
                                    }
                                }
                            }
                            else if (bitsAllocated == 8)
                            {
                                // 8-bit data
                                auto pixelValues = static_cast<const uint8_t*>(data);
                                for (unsigned long i = 0; i < width * height; i++)
                                {
                                    float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope +
                                        rescaleIntercept;
                                    pixelData.push_back(hounsfield);
                                }
                            }
                            else
                            {
                                throw std::runtime_error("Unsupported bit depth: " + std::to_string(bitsAllocated));
                            }
                        }
                        else
                        {
                            // For label data, just copy the values without conversion
                            if (bitsAllocated == 16)
                            {
                                auto pixelValues = static_cast<const uint16_t*>(data);
                                for (unsigned long i = 0; i < width * height; i++)
                                {
                                    pixelData.push_back(pixelValues[i]);
                                }
                            }
                            else if (bitsAllocated == 8)
                            {
                                auto pixelValues = static_cast<const uint8_t*>(data);
                                for (unsigned long i = 0; i < width * height; i++)
                                {
                                    pixelData.push_back(pixelValues[i]);
                                }
                            }
                            else
                            {
                                throw std::runtime_error("Unsupported bit depth: " + std::to_string(bitsAllocated));
                            }
                        }
                    }
                }
            }
        }
    }

    // Create torch tensor from the pixel data
    torch::TensorOptions options;
    if (isLabel)
    {
        options = torch::TensorOptions().dtype(torch::kLong);
    }
    else
    {
        options = torch::TensorOptions().dtype(torch::kFloat32);
    }

    auto tensor = torch::from_blob(pixelData.data(),
                                   {
                                       1, static_cast<long>(depth),
                                       static_cast<long>(height),
                                       static_cast<long>(width)
                                   },
                                   options).clone();

    return tensor;
}


std::vector<std::pair<std::string, std::string>> MonaiDataLoader::loadFromConfig()
{
    std::vector<std::pair<std::string, std::string>> dataPairs;

    // Determine which data split to use
    std::string dataKey = trainMode_ ? "training" : "validation";
    if (!config_.contains(dataKey))
    {
        // Try alternative keys
        if (config_.contains("train") && trainMode_)
        {
            dataKey = "train";
        }
        else if (config_.contains("val") && !trainMode_)
        {
            dataKey = "val";
        }
        else if (config_.contains("test") && !trainMode_)
        {
            dataKey = "test";
        }
        else
        {
            return dataPairs; // Empty list
        }
    }

    // Extract data pairs
    if (config_.contains(dataKey))
    {
        for (auto& item : config_[dataKey])
        {
            std::string imagePath, labelPath;

            // Handle different config formats
            if (item.contains("image") && item.contains("label"))
            {
                imagePath = item["image"].get<std::string>();
                labelPath = item["label"].get<std::string>();
            }
            else if (item.is_object() && item.size() >= 2)
            {
                // Try to infer which keys are for images vs labels
                for (auto& [key, value] : item.items())
                {
                    if (key.find("image") != std::string::npos ||
                        key.find("img") != std::string::npos)
                    {
                        imagePath = value.get<std::string>();
                    }
                    else if (key.find("label") != std::string::npos ||
                        key.find("seg") != std::string::npos ||
                        key.find("mask") != std::string::npos)
                    {
                        labelPath = value.get<std::string>();
                    }
                }
            }

            if (!imagePath.empty() && !labelPath.empty())
            {
                // Resolve paths - they might be relative to dataset directory
                if (imagePath.substr(0, 1) != "/")
                {
                    imagePath = (bundlePath_ / imagePath).string();
                }

                if (labelPath.substr(0, 1) != "/")
                {
                    labelPath = (bundlePath_ / labelPath).string();
                }

                dataPairs.push_back({imagePath, labelPath});
            }
        }
    }

    return dataPairs;
}

std::vector<std::pair<std::string, std::string>> MonaiDataLoader::scanForImageLabelPairs()
{
    std::vector<std::pair<std::string, std::string>> dataPairs;

    std::cout << "Scanning for data directories..." << std::endl;

    // Common folder structures in MONAI bundles
    std::vector<std::filesystem::path> possibleDataDirs =
    {
        bundlePath_ / "data",
        bundlePath_ / "dataset",
        bundlePath_ / "imagesTr",
        bundlePath_ / "labelsTr",
        bundlePath_ / "DICOM"
    };

    // Scan for image/label pairs using common naming conventions
    for (const auto& dataDir : possibleDataDirs)
    {
        if (exists(dataDir) && is_directory(dataDir))
        {
            std::vector<std::pair<std::string, std::string>> dirPairs =
                scanDirectoryForImageLabelPairs(dataDir);

            // Add the found pairs to our collection
            dataPairs.insert(dataPairs.end(), dirPairs.begin(), dirPairs.end());
        }
    }

    return dataPairs;
}


std::vector<std::pair<std::string, std::string>> MonaiDataLoader::scanDirectoryForImageLabelPairs(
    const std::filesystem::path& directory)
{
    std::vector<std::pair<std::string, std::string>> dataPairs;

    // Collect all potential image and label files
    std::vector<std::filesystem::path> imageFiles;
    std::vector<std::filesystem::path> labelFiles;
    std::vector<std::filesystem::path> dicomDirs;

    // Common medical image formats
    // std::vector<std::string> imageExtensions = {
    //   ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".pt", ".npz", ".png", ".jpg", ".jpeg", ".dcm"
    //};

    //TODO
    std::vector<std::string> imageExtensions =
    {

        ".dcm"
    };


    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory))
    {
        if (entry.is_directory())
        {
            // Check if this might be a DICOM series directory
            bool hasDicom = false;
            for (const auto& fileEntry : std::filesystem::directory_iterator(entry.path()))
            {
                if (fileEntry.is_regular_file())
                {
                    DcmFileFormat fileFormat;
                    if (fileFormat.loadFile(fileEntry.path().string().c_str()).good())
                    {
                        hasDicom = true;
                        break;
                    }
                }
            }

            if (hasDicom)
            {
                // Try to determine if this is an image or label directory
                auto dirName = entry.path().filename().string();
                bool isLabel = false;

                for (const auto& labelWord : {"label", "seg", "mask", "gt"})
                {
                    if (dirName.find(labelWord) != std::string::npos)
                    {
                        isLabel = true;
                        break;
                    }
                }

                if (isLabel)
                {
                    labelFiles.push_back(entry.path());
                }
                else
                {
                    imageFiles.push_back(entry.path());
                }
            }

            continue;
        }

        if (!entry.is_regular_file()) continue;

        auto filename = entry.path().filename().string();
        auto extension = entry.path().extension().string();

        // Check if this is a supported image format
        if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end())
        {
            // For DICOM files, check if valid
            if (extension == ".dcm")
            {
                DcmFileFormat fileFormat;
                if (!fileFormat.loadFile(entry.path().string().c_str()).good())
                {
                    continue;
                }
            }

            // Try to determine if this is an image or label based on filename/path
            bool isLabel = false;

            // Check filename and path for label indicators
            std::vector<std::string> labelIndicators = {
                "label", "seg", "mask", "gt", "segmentation", "annotation"
            };

            for (const auto& indicator : labelIndicators)
            {
                if (filename.find(indicator) != std::string::npos ||
                    entry.path().string().find(indicator) != std::string::npos ||
                    entry.path().parent_path().filename().string().find(indicator) != std::string::npos)
                {
                    isLabel = true;
                    break;
                }
            }

            // Special case for train/val folders structure
            if (entry.path().parent_path().filename().string() == "labelsTr" ||
                entry.path().parent_path().filename().string() == "labelsTs")
            {
                isLabel = true;
            }

            if (isLabel)
            {
                labelFiles.push_back(entry.path());
            }
            else
            {
                imageFiles.push_back(entry.path());
            }
        }
    }

    // Try to match image and label files based on filename patterns
    for (const auto& imagePath : imageFiles)
    {
        std::string baseFilename = imagePath.stem().string();

        // For directories (like DICOM series), use the directory name
        if (is_directory(imagePath))
        {
            baseFilename = imagePath.filename().string();
        }

        // Remove common suffixes
        std::regex suffixPattern("_image$|_img$|_0+$");
        baseFilename = std::regex_replace(baseFilename, suffixPattern, "");

        // Find a matching label file
        for (const auto& labelPath : labelFiles)
        {
            std::string labelFilename = labelPath.stem().string();

            // For directories, use the directory name
            if (is_directory(labelPath))
            {
                labelFilename = labelPath.filename().string();
            }

            std::regex labelSuffixPattern("_label$|_seg$|_mask$|_gt$");
            labelFilename = std::regex_replace(labelFilename, labelSuffixPattern, "");

            // Check if the base filenames match or are substrings
            if (labelFilename == baseFilename ||
                labelFilename.find(baseFilename) != std::string::npos ||
                baseFilename.find(labelFilename) != std::string::npos)
            {
                dataPairs.push_back({imagePath.string(), labelPath.string()});
                break;
            }
        }
    }

    return dataPairs;
}
