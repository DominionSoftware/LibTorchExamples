#include "MONAIDataLoader.h"

#include <regex>
#include <tuple>
#include <tuple>
#include <tuple>
#include <tuple>
#include <torch/torch.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include "Vector3D.h"


using namespace torch_explorer;

MonaiDataLoader::MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode) : bundlePath_(bundlePath),
    trainMode_(trainMode)
{
    // Load metadata and configuration
    loadMetadata();
    loadPreprocessingConfig();
    std::cout << "MONAI DiNTS model loader initialized for " << (trainMode ? "training" : "inference") << std::endl;
}

void MonaiDataLoader::loadMetadata()
{
    std::vector<std::filesystem::path> metadataPaths =
    {
        bundlePath_ / "configs" / "metadata.json",
        bundlePath_ / "metadata.json"
    };

    for (const auto& path : metadataPaths)
    {
        if (exists(path))
        {
            std::ifstream metadataFile(path);
            metadataFile >> metadata_;
            std::cout << "Loaded model metadata from: " << path.string() << std::endl;

            // Extract key model parameters from metadata
            if (metadata_.contains("network_data_format"))
            {
                auto& netFormat = metadata_["network_data_format"];

                // Extract input parameters
                if (netFormat.contains("inputs") && netFormat["inputs"].contains("image"))
                {
                    auto& input = netFormat["inputs"]["image"];
                    modelParams_.inputChannels = input.value("num_channels", 1);
                    modelParams_.patchSize = {96, 96, 96}; // Default patch size

                    if (input.contains("spatial_shape") && input["spatial_shape"].is_array())
                    {
                        auto& shape = input["spatial_shape"];
                        if (shape.size() >= 3)
                        {
                            modelParams_.patchSize[0] = shape[0].get<int>();
                            modelParams_.patchSize[1] = shape[1].get<int>();
                            modelParams_.patchSize[2] = shape[2].get<int>();
                        }
                    }

                    if (input.contains("value_range") && input["value_range"].is_array())
                    {
                        auto& range = input["value_range"];
                        if (range.size() >= 2)
                        {
                            modelParams_.inputRange[0] = range[0].get<float>();
                            modelParams_.inputRange[1] = range[1].get<float>();
                        }
                    }
                }

                // Extract output parameters
                if (netFormat.contains("outputs") && netFormat["outputs"].contains("pred"))
                {
                    auto& output = netFormat["outputs"]["pred"];
                    modelParams_.outputClasses = output.value("num_channels", 3);
                }
            }

            break;
        }
    }
}

// Helper function to resolve variable references (e.g., @patch_size)
template <typename T>
T MonaiDataLoader::resolveReference(const nlohmann::json& config, const std::string& refName)
{
    // Remove the @ prefix
    std::string varName = refName.substr(1);

    // Check if the variable exists in the config
    if (config.contains(varName))
    {
        return config[varName].get<T>();
    }

    // If not found, throw an exception or return a default value
    std::cerr << "Warning: Reference not found: " << refName << ", using default value." << std::endl;
    if constexpr (std::is_same_v<T, int>) return 0;
    else if constexpr (std::is_same_v<T, float>) return 0.0f;
    else if constexpr (std::is_same_v<T, std::string>) return "";
    else return T{};
}

// Function to evaluate simple expressions
float MonaiDataLoader::evaluateExpression(const nlohmann::json& config, const std::string& expr)
{
    // Remove the $ prefix
    std::string expression = expr.substr(1);

    // Special case: "1.0 - float(@out_size) / float(@patch_size)"
    std::regex outSizeDivPatchSize(R"(1\.0\s*-\s*float\(@out_size\)\s*/\s*float\(@patch_size\))");
    if (std::regex_match(expression, outSizeDivPatchSize))
    {
        int outSize = resolveReference<int>(config, "@out_size");
        int patchSize = resolveReference<int>(config, "@patch_size");
        return 1.0f - static_cast<float>(outSize) / static_cast<float>(patchSize);
    }

    // Handle integer division with // (Python-style)
    std::regex intDivision(R"(\(\(@([a-zA-Z0-9_]+)\s*-\s*@([a-zA-Z0-9_]+)\)\s*\/\/\s*(\d+)\))");
    std::smatch match;
    if (std::regex_search(expression, match, intDivision))
    {
        std::string var1 = "@" + match[1].str();
        std::string var2 = "@" + match[2].str();
        int divisor = std::stoi(match[3].str());

        int val1 = resolveReference<int>(config, var1);
        int val2 = resolveReference<int>(config, var2);

        return static_cast<float>((val1 - val2) / divisor);
    }

    // For complex expressions like "((@patch_size - @out_size) // 2,) * 4"
    // Extract just the numbers and do a basic calculation
    if (expression.find("@patch_size") != std::string::npos &&
        expression.find("@out_size") != std::string::npos)
    {
        int patchSize = resolveReference<int>(config, "@patch_size");
        int outSize = resolveReference<int>(config, "@out_size");

        // For the specific expression "((@patch_size - @out_size) // 2,) * 4"
        if (expression.find("// 2") != std::string::npos && expression.find("* 4") != std::string::npos)
        {
            return ((patchSize - outSize) / 2); // The * 4 is handled separately
        }
    }

    // Default case if we can't parse
    std::cerr << "Warning: Unable to evaluate expression: " << expr << ", using 0.0" << std::endl;
    return 0.0f;
}

// Function to get a value, resolving references and expressions
template <typename T>
T MonaiDataLoader::getValue(const nlohmann::json& config, const nlohmann::json& value)
{
    // Check if it's a string that might be a reference or expression
    if (value.is_string())
    {
        auto strValue = value.get<std::string>();

        // Handle variable references (@variable)
        if (strValue.size() > 1 && strValue[0] == '@')
        {
            return resolveReference<T>(config, strValue);
        }
        // Handle expressions ($expression)
        if (strValue.size() > 1 && strValue[0] == '$')
        {
            if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, float>)
            {
                return evaluateExpression(config, strValue);
            }
            else if constexpr (std::is_integral_v<T> || std::is_same_v<T, int>)
            {
                return static_cast<T>(evaluateExpression(config, strValue));
            }
            else
            {
                std::cerr << "Warning: Expression evaluation to this type not supported, using default" << std::endl;
                return T{};
            }
        }
    }

    // Direct conversion for non-string or non-reference values
    return value.get<T>();
}

void MonaiDataLoader::loadPreprocessingConfig()
{
    std::vector<std::filesystem::path> configPaths =
    {
        bundlePath_ / "configs" / "inference.json",
        bundlePath_ / "configs" / "train.json"
    };

    for (const auto& path : configPaths)
    {
        if (exists(path))
        {
            try
            {
                std::ifstream configFile(path);
                nlohmann::json config;
                configFile >> config;

                // Look for preprocessing parameters in the config
                if (config.contains("preprocessing") && config["preprocessing"].contains("transforms"))
                {
                    auto& transforms = config["preprocessing"]["transforms"];

                    // Find ScaleIntensityRanged transform
                    for (auto& transform : transforms)
                    {
                        if (transform.contains("_target_") &&
                            transform["_target_"].get<std::string>().find("ScaleIntensityRanged") != std::string::npos)
                        {
                            preprocessing_.a_min = getValue<float>(config, transform.value("a_min", -87.0f));
                            preprocessing_.a_max = getValue<float>(config, transform.value("a_max", 199.0f));
                            preprocessing_.b_min = getValue<float>(config, transform.value("b_min", 0.0f));
                            preprocessing_.b_max = getValue<float>(config, transform.value("b_max", 1.0f));
                            preprocessing_.clip = getValue<bool>(config, transform.value("clip", true));

                            std::cout << "Found intensity scaling parameters: " << preprocessing_.a_min << " to "
                                << preprocessing_.a_max << " -> " << preprocessing_.b_min << " to "
                                << preprocessing_.b_max << std::endl;
                            break;
                        }
                    }
                }

                // Look for inference parameters using the enhanced parser
                if (config.contains("inferer"))
                {
                    auto& inferer = config["inferer"];

                    // Parse roi_size with enhanced handling for references
                    if (inferer.contains("roi_size"))
                    {
                        // If roi_size is a direct array
                        if (inferer["roi_size"].is_array())
                        {
                            auto& roiSize = inferer["roi_size"];
                            if (roiSize.size() >= 3)
                            {
                                modelParams_.patchSize[0] = getValue<int>(config, roiSize[0]);
                                modelParams_.patchSize[1] = getValue<int>(config, roiSize[1]);
                                modelParams_.patchSize[2] = getValue<int>(config, roiSize[2]);
                            }
                        }
                        // If roi_size is a reference (e.g., "@patch_size")
                        else if (inferer["roi_size"].is_string())
                        {
                            auto roiSizeRef = inferer["roi_size"].get<std::string>();

                            if (roiSizeRef.size() > 1 && roiSizeRef[0] == '@')
                            {
                                // Resolve the reference
                                if (config.contains(roiSizeRef.substr(1)))
                                {
                                    auto& resolvedValue = config[roiSizeRef.substr(1)];

                                    // Handle the resolved value based on its type
                                    if (resolvedValue.is_array() && resolvedValue.size() >= 3)
                                    {
                                        modelParams_.patchSize[0] = resolvedValue[0].get<int>();
                                        modelParams_.patchSize[1] = resolvedValue[1].get<int>();
                                        modelParams_.patchSize[2] = resolvedValue[2].get<int>();
                                    }
                                    else if (resolvedValue.is_number())
                                    {
                                        // If it's a single number, use it for all dimensions
                                        int size = resolvedValue.get<int>();
                                        modelParams_.patchSize[0] = size;
                                        modelParams_.patchSize[1] = size;
                                        modelParams_.patchSize[2] = size;
                                    }
                                }
                            }
                        }
                    }

                    // Parse overlap with enhanced handling for expressions
                    if (inferer.contains("overlap"))
                    {
                        // Direct value
                        if (inferer["overlap"].is_number())
                        {
                            modelParams_.overlapRatio = inferer["overlap"].get<float>();
                        }
                        // Reference or expression
                        else if (inferer["overlap"].is_string())
                        {
                            auto overlapValue = inferer["overlap"].get<std::string>();

                            // Handle reference (@variable)
                            if (overlapValue.size() > 1 && overlapValue[0] == '@')
                            {
                                if (config.contains(overlapValue.substr(1)))
                                {
                                    modelParams_.overlapRatio = config[overlapValue.substr(1)].get<float>();
                                }
                            }
                            // Handle expression ($expression)
                            else if (overlapValue.size() > 1 && overlapValue[0] == '$')
                            {
                                modelParams_.overlapRatio = evaluateExpression(config, overlapValue);
                            }
                        }
                    }

                    // Parse sw_batch_size
                    if (inferer.contains("sw_batch_size"))
                    {
                        modelParams_.swBatchSize = getValue<int>(config, inferer["sw_batch_size"]);
                    }

                    // Parse padding_mode
                    if (inferer.contains("padding_mode"))
                    {
                        modelParams_.paddingMode = getValue<std::string>(config, inferer["padding_mode"]);
                    }

                    // Parse padding value (cval)
                    if (inferer.contains("cval"))
                    {
                        modelParams_.paddingValue = getValue<float>(config, inferer["cval"]);
                    }

                    // Parse progress indicator
                    if (inferer.contains("progress"))
                    {
                        modelParams_.showProgress = getValue<bool>(config, inferer["progress"]);
                    }

                    // Handle extra_input_padding (complex case)
                    if (inferer.contains("extra_input_padding"))
                    {
                        auto paddingExpr = inferer["extra_input_padding"].get<std::string>();
                        if (paddingExpr.size() > 1 && paddingExpr[0] == '$')
                        {
                            float paddingValue = evaluateExpression(config, paddingExpr);

                            // Check if it's multiplied by 4 (for all dimensions)
                            if (paddingExpr.find("* 4") != std::string::npos)
                            {
                                modelParams_.extraPadding[0] = static_cast<int>(paddingValue);
                                modelParams_.extraPadding[1] = static_cast<int>(paddingValue);
                                modelParams_.extraPadding[2] = static_cast<int>(paddingValue);
                            }
                            else
                            {
                                modelParams_.extraPadding[0] = static_cast<int>(paddingValue);
                            }
                        }
                    }
                }

                // Successfully parsed configuration
                std::cout << "Loaded preprocessing config from: " << path.string() << std::endl;
                break;
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error parsing " << path.string() << ": " << e.what() << std::endl;
            }
        }
    }

    // Use default values if not found in config
    if (preprocessing_.a_min == preprocessing_.a_max)
    {
        std::cout << "Using default preprocessing parameters for pancreas CT" << std::endl;
        preprocessing_.a_min = -87.0f;
        preprocessing_.a_max = 199.0f;
        preprocessing_.b_min = 0.0f;
        preprocessing_.b_max = 1.0f;
        preprocessing_.clip = true;
    }

    // Print the parsed parameters for verification
    std::cout << "Configuration parameters:" << std::endl;
    std::cout << "  Patch Size: [" << modelParams_.patchSize[0] << ", "
        << modelParams_.patchSize[1] << ", " << modelParams_.patchSize[2] << "]" << std::endl;
    std::cout << "  Overlap Ratio: " << modelParams_.overlapRatio << std::endl;
    std::cout << "  SW Batch Size: " << modelParams_.swBatchSize << std::endl;
    std::cout << "  Padding Mode: " << modelParams_.paddingMode << std::endl;
    std::cout << "  Padding Value: " << modelParams_.paddingValue << std::endl;
    std::cout << "  Show Progress: " << (modelParams_.showProgress ? "true" : "false") << std::endl;
    std::cout << "  Extra Padding: [" << modelParams_.extraPadding[0] << ", "
        << modelParams_.extraPadding[1] << ", " << modelParams_.extraPadding[2] << "]" << std::endl;
}

torch::jit::Module MonaiDataLoader::loadBaseModel()
{
    std::vector<std::filesystem::path> modelPaths = {
        // bundlePath_ / "models" / "model_pancreas_ct_dints_segmentation.pt", // we want the torch script model.
        bundlePath_ / "models" / "model_pancreas_ct_dints_segmentation.ts"
    };

    for (const auto& path : modelPaths)
    {
        if (exists(path))
        {
            try
            {
                std::cout << "Loading DiNTS base model from: " << path.string() << std::endl;
                torch::jit::Module module = torch::jit::load(path.string());
                trainMode_ ? module.train() : module.eval();
                return module;
            }
            catch (const c10::Error& e)
            {
                std::cerr << "Error loading the model: " << e.what() << std::endl;
                throw std::runtime_error("Failed to load model from " + path.string());
            }
        }
    }

    throw std::runtime_error("Could not find a valid model file in the bundle");
}

std::vector<torch::Tensor> MonaiDataLoader::loadDicomStudy(const std::filesystem::path& folderPath)
{
    std::vector<torch::Tensor> volumes;

    // Check if this is a directory containing DICOM files
    if (!std::filesystem::is_directory(folderPath))
    {
        throw std::runtime_error("Expected a directory containing DICOM files: " + folderPath.string());
    }

    // Collect all DICOM files
    std::vector<std::string> dicomFiles;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            DcmFileFormat fileFormat;
            if (fileFormat.loadFile(entry.path().string().c_str()).good()) {
                dicomFiles.push_back(entry.path().string());
            }
        }
    }

    if (dicomFiles.empty())
    {
        throw std::runtime_error("No valid DICOM files found in: " + folderPath.string());
    }

    // Sort files by position
    sortDicomFilesByPosition(dicomFiles);

    // Load the volume
    torch::Tensor volume = loadDicomVolume(dicomFiles);

    // Apply preprocessing for the DiNTS model
    volume = preprocessVolume(volume);

    volumes.push_back(volume);
    return volumes;
}


std::tuple<Vector3D<double>,Vector3D<double>, Vector3D<double>> MonaiDataLoader::getImageOrientation(const std::string& file)
{
    OFVector<Float64> imageOrientationPatient;


    DcmFileFormat fileFormat;
    if (!fileFormat.loadFile(file.c_str()).good())
    {
        throw std::runtime_error("Unable to load DICOM file. " + file);
    }

    DcmElement* ele;

    DcmDataset* dataSet = fileFormat.getDataset();
    if (!dataSet->findAndGetElement(DCM_ImageOrientationPatient, ele).good())
    {
        throw std::runtime_error("Image Orientation Patient not valid.");
    }

    DcmDecimalString* dcmDs = dynamic_cast<DcmDecimalString*>(ele);
    if (dcmDs == nullptr)
    {
        throw std::runtime_error("Image Orientation Patient not valid.");

    }

    if (dcmDs->getFloat64Vector(imageOrientationPatient).good() && imageOrientationPatient.size() != 6)
    {
        throw std::runtime_error("Image Orientation Patient not valid.");
    }

        

    Vector3D<double> xVector;

    xVector[0] = imageOrientationPatient[0];
    xVector[1] = imageOrientationPatient[1];
    xVector[2] = imageOrientationPatient[2];

    Vector3D<double> yVector;

    yVector[0] = imageOrientationPatient[3];
    yVector[1] = imageOrientationPatient[4];
    yVector[2] = imageOrientationPatient[5];

    Vector3D<double> zVector = Vector3D<double>::cross3D(xVector, yVector);
    std::cout << xVector << std::endl;
    std::cout << yVector << std::endl;
    std::cout << zVector << std::endl;

    return std::make_tuple(xVector, yVector, zVector);
  
}


void MonaiDataLoader::sortDicomFilesByPosition(const std::vector<std::string>& dicomFiles)
{
    // Get orientation
    auto imageOrientation = getImageOrientation(dicomFiles[0]);


    for (auto& f : dicomFiles)
    {
        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(f.c_str()).good())
        {
            throw std::runtime_error("Unable to load DICOM file. " + f);
        }
        DcmElement* ele;
        DcmDataset* dataSet = fileFormat.getDataset();
        if (fileFormat.getDataset()->findAndGetElement(DCM_ImagePositionPatient, ele).good())
        {
            OFVector<Float64> imagePositionPatient;
            DcmDecimalString* dcmDs = dynamic_cast<DcmDecimalString*>(ele);
            if (dcmDs != nullptr)
            {
                if (dcmDs->getFloat64Vector(imagePositionPatient).good())
                {
                    
                    std::cout << imagePositionPatient[0] <<"," << imagePositionPatient[1] << "," << imagePositionPatient[2] << std::endl;
                    
                }
            }

        }
        else
        {
            throw std::runtime_error("Unable to load DICOM file. " + f);
        }
    }
            
}

torch::Tensor MonaiDataLoader::loadDicomVolume(const std::vector<std::string>& dicomFiles)
{
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

                        // For image data, convert to Hounsfield Units if CT
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
                }
            }
        }
    }

    // Create torch tensor from the pixel data
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    auto tensor = torch::from_blob(pixelData.data(),
                                   {
                                       1, // Batch dimension
                                       1, // Channel dimension (for MONAI models)
                                       static_cast<long>(depth),
                                       static_cast<long>(height),
                                       static_cast<long>(width)
                                   },
                                   options).clone();

    return tensor;
}

torch::Tensor MonaiDataLoader::preprocessVolume(torch::Tensor volume)
{
    // Apply intensity scaling according to MONAI preprocessing parameters
    volume = applyIntensityScaling(volume);

    // Ensure the volume has the correct data type
    volume = volume.to(torch::kFloat32);

    return volume;
}

torch::Tensor MonaiDataLoader::applyIntensityScaling(const torch::Tensor& tensor)
{
    // Implement the ScaleIntensityRanged transform from MONAI
    // a_min, a_max: intensity original range
    // b_min, b_max: intensity target range

    float a_min = preprocessing_.a_min;
    float a_max = preprocessing_.a_max;
    float b_min = preprocessing_.b_min;
    float b_max = preprocessing_.b_max;

    // Clip values to [a_min, a_max]
    torch::Tensor clipped;
    if (preprocessing_.clip)
    {
        clipped = clamp(tensor, a_min, a_max);
    }
    else
    {
        clipped = tensor;
    }

    // Scale to [b_min, b_max]
    float scale = (b_max - b_min) / (a_max - a_min);
    torch::Tensor scaled = b_min + scale * (clipped - a_min);

    return scaled;
}

torch::Tensor MonaiDataLoader::extractFeatures(torch::jit::Module& model, torch::Tensor& volume)
{
    torch::NoGradGuard no_grad;

    // Determine device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA for feature extraction" << std::endl;
    }

    volume = volume.to(device);
    model.to(device);

    // Use sliding window inference for large volumes
    torch::Tensor features;

    try
    {
        // For DiNTS model, we need to extract features from an intermediate layer
        // This would typically require modifying the model or using hooks
        // For now, we'll just run inference and use the output as features

        // Convert volume to patches if needed
        std::vector<torch::Tensor> patches = extractPatches(volume, modelParams_.patchSize);
        std::vector<torch::Tensor> patchFeatures;

        for (auto& patch : patches)
        {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(patch);

            // Run inference 
            torch::Tensor output = model.forward(inputs).toTensor();

            // For transfer learning, we typically want features before the final classification layer
            // For simplicity, we'll use the logits (before softmax) as features
            patchFeatures.push_back(output);
        }

        // Combine patch features
        features = cat(patchFeatures, 0);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error during feature extraction: " << e.what() << std::endl;
        throw std::runtime_error("Failed to extract features from the model");
    }

    return features;
}

std::vector<torch::Tensor> MonaiDataLoader::extractPatches(const torch::Tensor& volume,
                                                           const std::vector<int>& patchSize)
{
    std::vector<torch::Tensor> patches;

    // Get volume dimensions
    auto dims = volume.sizes();
    int depth = dims[2];
    int height = dims[3];
    int width = dims[4];

    // Calculate stride with overlap
    int stride_z = static_cast<int>(patchSize[0] * (1 - modelParams_.overlapRatio));
    int stride_y = static_cast<int>(patchSize[1] * (1 - modelParams_.overlapRatio));
    int stride_x = static_cast<int>(patchSize[2] * (1 - modelParams_.overlapRatio));

    // Extract patches
    for (int z = 0; z <= depth - patchSize[0]; z += stride_z)
    {
        for (int y = 0; y <= height - patchSize[1]; y += stride_y)
        {
            for (int x = 0; x <= width - patchSize[2]; x += stride_x)
            {
                // Extract patch
                torch::Tensor patch = volume.slice(2, z, z + patchSize[0])
                                            .slice(3, y, y + patchSize[1])
                                            .slice(4, x, x + patchSize[2]);

                patches.push_back(patch);
            }
        }
    }

    // If no patches were extracted (volume smaller than patch size), resize the volume
    if (patches.empty())
    {
        torch::Tensor resized = interpolate(
            volume,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{patchSize[0], patchSize[1], patchSize[2]})
            .mode(torch::kTrilinear)
            .align_corners(false)
        );

        patches.push_back(resized);
    }

    return patches;
}




