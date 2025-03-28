#include "MONAIDataLoader.h"
#include <torch/torch.h>

using namespace torch_explorer;

MonaiDataLoader::MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode)
    : bundlePath_(bundlePath), trainMode_(trainMode)
{
    // Load metadata and configuration
    loadMetadata();
    loadPreprocessingConfig();
    std::cout << "MONAI DiNTS model loader initialized for " << (trainMode ? "training" : "inference") << std::endl;
}

void MonaiDataLoader::loadMetadata()
{
    std::vector<std::filesystem::path> metadataPaths = {
        bundlePath_ / "configs" / "metadata.json",
        bundlePath_ / "metadata.json"
    };

    metadata_ = jsonLoader_.loadJSON(metadataPaths, "network_data_format");

    if (!metadata_.empty() && metadata_.contains("network_data_format")) {
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
}

void MonaiDataLoader::loadPreprocessingConfig()
{
    std::vector<std::filesystem::path> configPaths = {
        bundlePath_ / "configs" / "inference.json",
        bundlePath_ / "configs" / "train.json"
    };

    nlohmann::json config = jsonLoader_.loadJSON(configPaths);

    if (!config.empty()) {
        // Parse preprocessing parameters
        parsePreprocessingParameters(config);

        // Parse model parameters
        parseModelParameters(config);
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

void MonaiDataLoader::parsePreprocessingParameters(nlohmann::json& config)
{
    // Look for preprocessing parameters in the config
    if (config.contains("preprocessing") && config["preprocessing"].contains("transforms")) {
        auto& transforms = config["preprocessing"]["transforms"];

        // Find ScaleIntensityRanged transform
        for (auto& transform : transforms) {
            if (transform.contains("_target_") &&
                transform["_target_"].get<std::string>().find("ScaleIntensityRanged") != std::string::npos) {
                preprocessing_.a_min = jsonLoader_.getValue<float>(config, transform.value("a_min", -87.0f));
                preprocessing_.a_max = jsonLoader_.getValue<float>(config, transform.value("a_max", 199.0f));
                preprocessing_.b_min = jsonLoader_.getValue<float>(config, transform.value("b_min", 0.0f));
                preprocessing_.b_max = jsonLoader_.getValue<float>(config, transform.value("b_max", 1.0f));
                preprocessing_.clip = jsonLoader_.getValue<bool>(config, transform.value("clip", true));

                std::cout << "Found intensity scaling parameters: " << preprocessing_.a_min << " to "
                    << preprocessing_.a_max << " -> " << preprocessing_.b_min << " to "
                    << preprocessing_.b_max << std::endl;
                break;
            }
        }
    }
}

void MonaiDataLoader::parseModelParameters(nlohmann::json& config)
{
    // Look for inference parameters
    if (config.contains("inferer")) {
        auto& inferer = config["inferer"];

        // Parse roi_size with enhanced handling for references
        if (inferer.contains("roi_size")) {
            // If roi_size is a direct array
            if (inferer["roi_size"].is_array()) {
                auto& roiSize = inferer["roi_size"];
                if (roiSize.size() >= 3) {
                    modelParams_.patchSize[0] = jsonLoader_.getValue<int>(config, roiSize[0]);
                    modelParams_.patchSize[1] = jsonLoader_.getValue<int>(config, roiSize[1]);
                    modelParams_.patchSize[2] = jsonLoader_.getValue<int>(config, roiSize[2]);
                }
            }
            // If roi_size is a reference (e.g., "@patch_size")
            else if (inferer["roi_size"].is_string()) {
                auto roiSizeRef = inferer["roi_size"].get<std::string>();

                if (roiSizeRef.size() > 1 && roiSizeRef[0] == '@') {
                    // Resolve the reference
                    if (config.contains(roiSizeRef.substr(1))) {
                        auto& resolvedValue = config[roiSizeRef.substr(1)];

                        // Handle the resolved value based on its type
                        if (resolvedValue.is_array() && resolvedValue.size() >= 3) {
                            modelParams_.patchSize[0] = resolvedValue[0].get<int>();
                            modelParams_.patchSize[1] = resolvedValue[1].get<int>();
                            modelParams_.patchSize[2] = resolvedValue[2].get<int>();
                        }
                        else if (resolvedValue.is_number()) {
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
        if (inferer.contains("overlap")) {
            // Direct value
            if (inferer["overlap"].is_number()) {
                modelParams_.overlapRatio = inferer["overlap"].get<float>();
            }
            // Reference or expression
            else if (inferer["overlap"].is_string()) {
                auto overlapValue = inferer["overlap"].get<std::string>();

                // Handle reference (@variable)
                if (overlapValue.size() > 1 && overlapValue[0] == '@') {
                    if (config.contains(overlapValue.substr(1))) {
                        modelParams_.overlapRatio = config[overlapValue.substr(1)].get<float>();
                    }
                }
                // Handle expression ($expression)
                else if (overlapValue.size() > 1 && overlapValue[0] == '$') {
                    modelParams_.overlapRatio = jsonLoader_.evaluateExpression(config, overlapValue);
                }
            }
        }

        // Parse sw_batch_size
        if (inferer.contains("sw_batch_size")) {
            modelParams_.swBatchSize = jsonLoader_.getValue<int>(config, inferer["sw_batch_size"]);
        }

        // Parse padding_mode
        if (inferer.contains("padding_mode")) {
            modelParams_.paddingMode = jsonLoader_.getValue<std::string>(config, inferer["padding_mode"]);
        }

        // Parse padding value (cval)
        if (inferer.contains("cval")) {
            modelParams_.paddingValue = jsonLoader_.getValue<float>(config, inferer["cval"]);
        }

        // Parse progress indicator
        if (inferer.contains("progress")) {
            modelParams_.showProgress = jsonLoader_.getValue<bool>(config, inferer["progress"]);
        }

        // Handle extra_input_padding (complex case)
        if (inferer.contains("extra_input_padding")) {
            auto paddingExpr = inferer["extra_input_padding"].get<std::string>();
            if (paddingExpr.size() > 1 && paddingExpr[0] == '$') {
                float paddingValue = jsonLoader_.evaluateExpression(config, paddingExpr);

                // Check if it's multiplied by 4 (for all dimensions)
                if (paddingExpr.find("* 4") != std::string::npos) {
                    modelParams_.extraPadding[0] = static_cast<int>(paddingValue);
                    modelParams_.extraPadding[1] = static_cast<int>(paddingValue);
                    modelParams_.extraPadding[2] = static_cast<int>(paddingValue);
                }
                else {
                    modelParams_.extraPadding[0] = static_cast<int>(paddingValue);
                }
            }
        }
    }
}

torch::jit::Module MonaiDataLoader::loadBaseModel()
{
    std::vector<std::filesystem::path> modelPaths = {
        bundlePath_ / "models" / "model_pancreas_ct_dints_segmentation.ts"
    };

    for (const auto& path : modelPaths) {
        if (std::filesystem::exists(path)) {
            try {
                std::cout << "Loading DiNTS base model from: " << path.string() << std::endl;
                torch::jit::Module module = torch::jit::load(path.string());
                trainMode_ ? module.train() : module.eval();
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

std::vector<torch::Tensor> MonaiDataLoader::loadDicomStudy(const std::filesystem::path& folderPath)
{
    std::vector<torch::Tensor> volumes;

    // Use DicomLoader to load the volume
    torch::Tensor volume = dicomLoader_.loadDicomStudy(folderPath);

    // Apply preprocessing for the DiNTS model
    volume = preprocessVolume(volume);

    volumes.push_back(volume);
    return volumes;
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