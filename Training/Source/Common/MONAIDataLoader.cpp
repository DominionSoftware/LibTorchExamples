#include "MONAIDataLoader.h"

#include <regex>
#include <torch/torch.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>


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
        if (std::filesystem::exists(path))
        {
            std::ifstream metadataFile(path);
            metadataFile >> metadata_;
            std::cout << "Loaded model metadata from: " << path.string() << std::endl;

            // Extract key model parameters from metadata
            if (metadata_.contains("network_data_format")) {
                auto& netFormat = metadata_["network_data_format"];

                // Extract input parameters
                if (netFormat.contains("inputs") && netFormat["inputs"].contains("image")) {
                    auto& input = netFormat["inputs"]["image"];
                    modelParams_.inputChannels = input.value("num_channels", 1);
                    modelParams_.patchSize = { 96, 96, 96 }; // Default patch size

                    if (input.contains("spatial_shape") && input["spatial_shape"].is_array()) {
                        auto& shape = input["spatial_shape"];
                        if (shape.size() >= 3) {
                            modelParams_.patchSize[0] = shape[0].get<int>();
                            modelParams_.patchSize[1] = shape[1].get<int>();
                            modelParams_.patchSize[2] = shape[2].get<int>();
                        }
                    }

                    if (input.contains("value_range") && input["value_range"].is_array()) {
                        auto& range = input["value_range"];
                        if (range.size() >= 2) {
                            modelParams_.inputRange[0] = range[0].get<float>();
                            modelParams_.inputRange[1] = range[1].get<float>();
                        }
                    }
                }

                // Extract output parameters
                if (netFormat.contains("outputs") && netFormat["outputs"].contains("pred")) {
                    auto& output = netFormat["outputs"]["pred"];
                    modelParams_.outputClasses = output.value("num_channels", 3);
                }
            }

            break;
        }
    }
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
        if (std::filesystem::exists(path))
        {
            try {
                std::ifstream configFile(path);
                nlohmann::json config;
                configFile >> config;

                // Look for preprocessing parameters in the config
                if (config.contains("preprocessing") && config["preprocessing"].contains("transforms")) {
                    auto& transforms = config["preprocessing"]["transforms"];

                    // Find ScaleIntensityRanged transform
                    for (auto& transform : transforms) {
                        if (transform.contains("_target_") &&
                            transform["_target_"].get<std::string>().find("ScaleIntensityRanged") != std::string::npos) {

                            preprocessing_.a_min = transform.value("a_min", -87.0f);
                            preprocessing_.a_max = transform.value("a_max", 199.0f);
                            preprocessing_.b_min = transform.value("b_min", 0.0f);
                            preprocessing_.b_max = transform.value("b_max", 1.0f);
                            preprocessing_.clip = transform.value("clip", true);

                            std::cout << "Found intensity scaling parameters: " << preprocessing_.a_min << " to "
                                << preprocessing_.a_max << " -> " << preprocessing_.b_min << " to "
                                << preprocessing_.b_max << std::endl;
                            break;
                        }
                    }
                }

                // Look for inference parameters
                if (config.contains("inferer")) {
                    auto& inferer = config["inferer"];

                    if (inferer.contains("roi_size") && inferer["roi_size"].is_array()) {
                        auto& roiSize = inferer["roi_size"];
                        if (roiSize.size() >= 3) {
                            modelParams_.patchSize[0] = roiSize[0].get<int>();
                            modelParams_.patchSize[1] = roiSize[1].get<int>();
                            modelParams_.patchSize[2] = roiSize[2].get<int>();
                        }
                    }

                    if (inferer.contains("overlap")) {
                        modelParams_.overlapRatio = inferer["overlap"].get<float>();
                    }
                }

                // Successfully parsed configuration
                std::cout << "Loaded preprocessing config from: " << path.string() << std::endl;
                break;
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing " << path.string() << ": " << e.what() << std::endl;
            }
        }
    }

    // Use default values if not found in config
    if (preprocessing_.a_min == preprocessing_.a_max) {
        std::cout << "Using default preprocessing parameters for pancreas CT" << std::endl;
        preprocessing_.a_min = -87.0f;
        preprocessing_.a_max = 199.0f;
        preprocessing_.b_min = 0.0f;
        preprocessing_.b_max = 1.0f;
        preprocessing_.clip = true;
    }
}

torch::jit::Module MonaiDataLoader::loadBaseModel()
{
    // Look for model files in the actual locations based on your folder structure
    std::vector<std::filesystem::path> modelPaths = {
        bundlePath_ / "models" / "model_pancreas_ct_dints_segmentation.pt",
        bundlePath_ / "models" / "model_pancreas_ct_dints_segmentation.ts"
    };

    for (const auto& path : modelPaths)
    {
        if (std::filesystem::exists(path))
        {
            try {
                std::cout << "Loading DiNTS base model from: " << path.string() << std::endl;
                torch::jit::Module module = torch::jit::load(path.string());
                module.eval();  // Set to evaluation mode
                return module;
            }
            catch (const c10::Error& e) {
                std::cerr << "Error loading the model: " << e.what() << std::endl;
                throw std::runtime_error("Failed to load model from " + path.string());
            }
        }
    }

    throw std::runtime_error("Could not find a valid model file in the bundle");
}

std::vector<torch::Tensor> MonaiDataLoader::loadDicomStudy(const std::string& folderPath)
{
    std::vector<torch::Tensor> volumes;

    // Check if this is a directory containing DICOM files
    if (!std::filesystem::is_directory(folderPath)) {
        throw std::runtime_error("Expected a directory containing DICOM files: " + folderPath);
    }

    // Collect all DICOM files
    std::vector<std::string> dicomFiles;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dcm") {
            DcmFileFormat fileFormat;
            if (fileFormat.loadFile(entry.path().string().c_str()).good()) {
                dicomFiles.push_back(entry.path().string());
            }
        }
    }

    if (dicomFiles.empty()) {
        throw std::runtime_error("No valid DICOM files found in: " + folderPath);
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

void MonaiDataLoader::sortDicomFilesByPosition(std::vector<std::string>& dicomFiles)
{
    
            
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
            1,                      // Batch dimension
            1,                      // Channel dimension (for MONAI models)
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
    if (preprocessing_.clip) {
        clipped = torch::clamp(tensor, a_min, a_max);
    }
    else {
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
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA for feature extraction" << std::endl;
    }

    volume = volume.to(device);
    model.to(device);

    // Use sliding window inference for large volumes
    torch::Tensor features;

    try {
        // For DiNTS model, we need to extract features from an intermediate layer
        // This would typically require modifying the model or using hooks
        // For now, we'll just run inference and use the output as features

        // Convert volume to patches if needed
        std::vector<torch::Tensor> patches = extractPatches(volume, modelParams_.patchSize);
        std::vector<torch::Tensor> patchFeatures;

        for (auto& patch : patches) {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(patch);

            // Run inference 
            torch::Tensor output = model.forward(inputs).toTensor();

            // For transfer learning, we typically want features before the final classification layer
            // For simplicity, we'll use the logits (before softmax) as features
            patchFeatures.push_back(output);
        }

        // Combine patch features
        features = torch::cat(patchFeatures, 0);

    }
    catch (const c10::Error& e) {
        std::cerr << "Error during feature extraction: " << e.what() << std::endl;
        throw std::runtime_error("Failed to extract features from the model");
    }

    return features;
}

std::vector<torch::Tensor> MonaiDataLoader::extractPatches(const torch::Tensor& volume, const std::vector<int>& patchSize)
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
    for (int z = 0; z <= depth - patchSize[0]; z += stride_z) {
        for (int y = 0; y <= height - patchSize[1]; y += stride_y) {
            for (int x = 0; x <= width - patchSize[2]; x += stride_x) {
                // Extract patch
                torch::Tensor patch = volume.slice(2, z, z + patchSize[0])
                    .slice(3, y, y + patchSize[1])
                    .slice(4, x, x + patchSize[2]);

                patches.push_back(patch);
            }
        }
    }

    // If no patches were extracted (volume smaller than patch size), resize the volume
    if (patches.empty()) {
        torch::Tensor resized = torch::nn::functional::interpolate(
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

// Create a classifier model for transfer learning
torch::nn::Sequential MonaiDataLoader::createClassifier(int numFeatures, int numClasses)
{
    torch::nn::Sequential classifier(
        torch::nn::Linear(numFeatures, 256),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(256, 64),
        torch::nn::ReLU(),
        torch::nn::Dropout(0.3),
        torch::nn::Linear(64, numClasses)
    );

    return classifier;
}

// Save the trained classifier
void MonaiDataLoader::saveClassifier(const torch::nn::Sequential& classifier, const std::string& path)
{
    torch::save(classifier, path);
    std::cout << "Saved classifier to: " << path << std::endl;
}

// Load a previously trained classifier
torch::nn::Sequential MonaiDataLoader::loadClassifier(const std::string& path, int numFeatures, int numClasses)
{
    auto classifier = createClassifier(numFeatures, numClasses);
    torch::load(classifier, path);
    std::cout << "Loaded classifier from: " << path << std::endl;
    return classifier;
}
