#include "MONAIDataLoader.h"
#include <torch/torch.h>
#include <algorithm>

#include "FileSaver.h"
#include "vtkImageResample.h"
#include "vtkImageThreshold.h"
#include "vtkImageShiftScale.h"


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

    if (!metadata_.empty() && metadata_.contains("network_data_format"))
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
}

void MonaiDataLoader::loadPreprocessingConfig()
{
    std::vector<std::filesystem::path> configPaths = {
        bundlePath_ / "configs" / "inference.json",
        bundlePath_ / "configs" / "train.json"
    };

    nlohmann::json config = jsonLoader_.loadJSON(configPaths);

    if (!config.empty())
    {
        // Parse preprocessing parameters
        parsePreprocessingParameters(config);

        // Parse model parameters
        parseModelParameters(config);
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

void MonaiDataLoader::parsePreprocessingParameters(nlohmann::json& config)
{
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
                    modelParams_.patchSize[0] = jsonLoader_.getValue<int>(config, roiSize[0]);
                    modelParams_.patchSize[1] = jsonLoader_.getValue<int>(config, roiSize[1]);
                    modelParams_.patchSize[2] = jsonLoader_.getValue<int>(config, roiSize[2]);
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
                    modelParams_.overlapRatio = jsonLoader_.evaluateExpression(config, overlapValue);
                }
            }
        }

        // Parse sw_batch_size
        if (inferer.contains("sw_batch_size"))
        {
            modelParams_.swBatchSize = jsonLoader_.getValue<int>(config, inferer["sw_batch_size"]);
        }

        // Parse padding_mode
        if (inferer.contains("padding_mode"))
        {
            modelParams_.paddingMode = jsonLoader_.getValue<std::string>(config, inferer["padding_mode"]);
        }

        // Parse padding value (cval)
        if (inferer.contains("cval"))
        {
            modelParams_.paddingValue = jsonLoader_.getValue<float>(config, inferer["cval"]);
        }

        // Parse progress indicator
        if (inferer.contains("progress"))
        {
            modelParams_.showProgress = jsonLoader_.getValue<bool>(config, inferer["progress"]);
        }

        // Handle extra_input_padding (complex case)
        if (inferer.contains("extra_input_padding"))
        {
            auto paddingExpr = inferer["extra_input_padding"].get<std::string>();
            if (paddingExpr.size() > 1 && paddingExpr[0] == '$')
            {
                float paddingValue = jsonLoader_.evaluateExpression(config, paddingExpr);

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
}

torch::jit::Module MonaiDataLoader::loadBaseModel()
{
    std::vector<std::filesystem::path> modelPaths = {
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

torch::Tensor MonaiDataLoader::loadDicomStudy(const std::filesystem::path& folderPath)
{
    // Use DicomLoader to load the volume
     auto metaData = dicomLoader_.loadDicomStudy(folderPath);


     std::cout << metaData[0].pixelSpacing_[0] << "," << metaData[0].pixelSpacing_[0] << "," << metaData[0].pixelSpacing_[0] << std::endl;

     std::for_each(metaData.begin(), metaData.end(), [](DicomMetaData& md)
         {
             std::cout << md.imagePositionPatient_[2] << std::endl;
         });

     // Since we currently only allow Z axis slices, we sort on the Z position.
     if (DicomLoader::isLPS(metaData[0].imageOrientationPatient_))
     {
         std::ranges::sort(metaData, [](DicomMetaData& a, DicomMetaData& b)->bool
             {
                 return a.imagePositionPatient_[2] < b.imagePositionPatient_[2];
             }
         );
     }
     else
     {
         std::ranges::sort(metaData, [](DicomMetaData& a, DicomMetaData& b)->bool
             {
                 return b.imagePositionPatient_[2] < a.imagePositionPatient_[2];
             }
         );
     }
    
     std::for_each(metaData.begin(), metaData.end(), [](DicomMetaData& md)
         {
             std::cout << md.imagePositionPatient_[2] << std::endl;
         });

     // Compute the average z spacing...

     std::vector<double> spacings(metaData.size() - 1);

     // Calculate all adjacent differences
     std::transform(metaData.begin() + 1, metaData.end(), metaData.begin(), spacings.begin(),
         [](const DicomMetaData& current, const DicomMetaData& previous) -> double {
             return std::abs(current.imagePositionPatient_[2] - previous.imagePositionPatient_[2]);
         });

     // Calculate the average
     double sum = std::accumulate(spacings.begin(), spacings.end(), 0.0);
     double average = sum / spacings.size();
     std::cout << "average pixel spacing = " << average << std::endl;
     vtkSmartPointer<vtkImageData> image = dicomLoader_.loadToVTK(metaData,average);
     vtkSmartPointer<vtkImageData> rescaledImage = rescaleCT(image, 250, 3000);

     vtkSmartPointer<vtkImageData> resampledImage = resizeCT(rescaledImage, 1.5, { 96,96,96 });

     FileSaver saver("D:\\Projects\\Pancreas-CT.bin\\RelWithDebInfo");

     saver.saveAsMHA(resampledImage, "images", "scaledAndResampled.mha");

     

    //return volume;
     return torch::Tensor();

}

vtkSmartPointer<vtkImageData> MonaiDataLoader::resizeCT(vtkSmartPointer<vtkImageData> input, double spacing, std::array<int,3> size)
{

    int inputDims[3];
    input->GetDimensions(inputDims);
    double inputSpacing[3];
    input->GetSpacing(inputSpacing);
    double inputOrigin[3];
    input->GetOrigin(inputOrigin);
    double newSpacing[3] = { spacing, spacing, spacing };
    vtkSmartPointer<vtkImageResample> resample = vtkSmartPointer<vtkImageResample>::New();
    resample->SetInputData(input);
    resample->SetDimensionality(3);
    resample->SetOutputSpacing(newSpacing);
    double newOrigin[3];
    for (int i = 0; i < 3; i++) {
        // Center the resized volume
        newOrigin[i] = inputOrigin[i] + 0.5 * (inputDims[i] * inputSpacing[i] - size[i] * spacing);
    }

    resample->SetOutputOrigin(newOrigin);
    resample->SetOutputExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1);

    // Use linear interpolation for medical images
    resample->SetInterpolationModeToLinear();
    resample->Update();

    return resample->GetOutput();

}


vtkSmartPointer<vtkImageData> MonaiDataLoader::rescaleCT(vtkSmartPointer<vtkImageData> input, double min, double max)
{
    constexpr double airHU = -1000.0;

    vtkSmartPointer<vtkImageThreshold> thresholder = vtkSmartPointer<vtkImageThreshold>::New();
    thresholder->SetInputData(input);
    thresholder->ThresholdBetween(min, max);
    thresholder->SetOutValue(airHU);
    thresholder->SetInValue(0);
    thresholder->ReplaceInOff();           
    thresholder->ReplaceOutOn();           
    thresholder->SetOutputScalarTypeToFloat();
    vtkSmartPointer<vtkImageShiftScale> scaler = vtkSmartPointer<vtkImageShiftScale>::New();
    scaler->SetInputConnection(thresholder->GetOutputPort());
    scaler->SetOutputScalarTypeToFloat();
    scaler->SetShift(-airHU);
    scaler->SetScale(1.0 / (max - airHU));
    scaler->Update();

    return scaler->GetOutput();

}
torch::Tensor MonaiDataLoader::preprocessCTScan(vtkSmartPointer<vtkImageData> ctScan)
{

    vtkSmartPointer<vtkImageData> rescaledImage = rescaleCT(ctScan, 250, 3000);

    vtkSmartPointer<vtkImageData> resampledImage = resizeCT(rescaledImage, 1.5, { 96,96,96 });

   
    // 5. Create a torch tensor from the buffer
    int* dims = resampledImage->GetDimensions();
    float* buffer = static_cast<float*>(resampledImage->GetScalarPointer());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);


    torch::Tensor volumeTensor = torch::from_blob(
        buffer,
        { dims[2], dims[1], dims[0] },  // PyTorch uses [D, H, W] while VTK uses [W, H, D]
        options
    ).clone();


    volumeTensor = volumeTensor.unsqueeze(0).unsqueeze(0);  // Shape becomes [1, 1, D, H, W]

    volumeTensor = volumeTensor.contiguous();

    std::cout << "Final tensor shape: " << volumeTensor.sizes() << std::endl;

    return volumeTensor;
}



torch::Tensor MonaiDataLoader::inference(torch::Tensor& volume)
{
   
    return torch::Tensor();
}


torch::Tensor MonaiDataLoader::slidingWindowInference(torch::Tensor& volume, torch::jit::Module& model)
{
    auto volumeSize = volume.sizes().vec();
    std::int64_t batchSize = volumeSize[0];
    std::int64_t channels = volumeSize[1];
    std::int64_t depth = volumeSize[2];
    std::int64_t height = volumeSize[3];
    std::int64_t width = volumeSize[4];


    std::cout << batchSize << std::endl;
    std::cout << channels << std::endl;
    std::cout << depth << std::endl;
    std::cout << height << std::endl;
    std::cout << width << std::endl;


    const auto& patchSize = modelParams_.patchSize;
    int stride_z = static_cast<int>(patchSize[0] * (1 - modelParams_.overlapRatio));
    int stride_y = static_cast<int>(patchSize[1] * (1 - modelParams_.overlapRatio));
    int stride_x = static_cast<int>(patchSize[2] * (1 - modelParams_.overlapRatio));
    stride_z = std::max<int>(1, stride_z);
    stride_y = std::max<int>(1, stride_y);
    stride_x = std::max<int>(1, stride_x);

    // Calculate padding needed to ensure full coverage
    std::int64_t pad_z = ((depth - 1) / stride_z + 1) * stride_z + patchSize[0] - depth;
    std::int64_t pad_y = ((height - 1) / stride_y + 1) * stride_y + patchSize[1] - height;
    std::int64_t pad_x = ((width - 1) / stride_x + 1) * stride_x + patchSize[2] - width;


    auto paddingOption = modelParams_.paddingMode;
    float paddingValue = modelParams_.paddingValue;

    torch::Tensor paddedVolume;
    if (pad_z > 0 || pad_y > 0 || pad_x > 0)
    {
        // Convert padding to format needed by pad function
        std::vector<int64_t> padding = {0, pad_x, 0, pad_y, 0, pad_z};

        // Apply padding
        paddedVolume = torch::nn::functional::pad(
            volume,
            torch::nn::functional::PadFuncOptions(padding).mode(torch::kConstant).value(paddingValue)
        );
    }
    else
    {
        paddedVolume = volume;
    }


    auto paddedSize = paddedVolume.sizes().vec();
    int paddedDepth = paddedSize[2];
    int paddedHeight = paddedSize[3];
    int paddedWidth = paddedSize[4];

    std::vector<torch::Tensor> patchResults;
    std::vector<std::array<int, 3>> patchPositions;
    std::vector<torch::Tensor> patchBatch;


    for (int z = 0; z <= paddedDepth - patchSize[0]; z += stride_z)
    {
        for (int y = 0; y <= paddedHeight - patchSize[1]; y += stride_y)
        {
            for (int x = 0; x <= paddedWidth - patchSize[2]; x += stride_x)
            {
                // Extract patch
                torch::Tensor patch = paddedVolume.slice(2, z, z + patchSize[0])
                                                  .slice(3, y, y + patchSize[1])
                                                  .slice(4, x, x + patchSize[2]);

                patchBatch.push_back(patch);
                patchPositions.push_back({z, y, x});

                // Process batch when it reaches sw_batch_size
                if (patchBatch.size() >= static_cast<size_t>(modelParams_.swBatchSize) ||
                    (z >= paddedDepth - patchSize[0] &&
                        y >= paddedHeight - patchSize[1] &&
                        x >= paddedWidth - patchSize[2]))
                {
                    // Combine patches into a batch
                    torch::Tensor batch = cat(patchBatch, 0);

                    // Run inference
                    std::vector<torch::jit::IValue> inputs;
                    inputs.push_back(batch);
                    torch::Tensor output = model.forward(inputs).toTensor();

                    // Split batch result back into individual patches
                    for (int i = 0; i < static_cast<int>(patchBatch.size()); i++)
                    {
                        patchResults.push_back(output[i].unsqueeze(0));
                    }

                    // Clear batch for next iteration
                    patchBatch.clear();
                }
            }
        }
    }

    // Stitch patches back together
    std::array<int, 3> originalSize = {depth, height, width};
    torch::Tensor result = stitchPatches(patchResults, patchPositions, originalSize);

    // Get argmax to create final segmentation
    result = argmax(result, 1, false);

    return result;

    return torch::Tensor();
}

torch::Tensor MonaiDataLoader::stitchPatches(const std::vector<torch::Tensor>& patches,
                                             const std::vector<std::array<int, 3>>& positions,
                                             const std::array<int, 3>& volumeSize)
{
    // Get output classes from the first patch
    int outputClasses = patches[0].size(1);

    // Create empty volume for result with probability maps
    torch::Tensor result = torch::zeros({1, outputClasses, volumeSize[0], volumeSize[1], volumeSize[2]},
                                        torch::kFloat32);

    // Create count tensor to average overlapping regions
    torch::Tensor count = torch::zeros({1, 1, volumeSize[0], volumeSize[1], volumeSize[2]},
                                       torch::kFloat32);

    const std::vector<int>& patchSize = modelParams_.patchSize;

    // Accumulate patches
    for (size_t i = 0; i < patches.size(); i++)
    {
        auto& patch = patches[i];
        auto& pos = positions[i];

        // Get effective patch size (might be smaller at boundaries)
        int effective_z = std::min<int>(patchSize[0], volumeSize[0] - pos[0]);
        int effective_y = std::min<int>(patchSize[1], volumeSize[1] - pos[1]);
        int effective_x = std::min<int>(patchSize[2], volumeSize[2] - pos[2]);

        // Slice the patch to effective size (handling boundary cases)
        torch::Tensor effectivePatch = patch.slice(2, 0, effective_z)
                                            .slice(3, 0, effective_y)
                                            .slice(4, 0, effective_x);

        // Add patch to result
        result.slice(2, pos[0], pos[0] + effective_z)
              .slice(3, pos[1], pos[1] + effective_y)
              .slice(4, pos[2], pos[2] + effective_x)
              .add_(effectivePatch);

        // Update count for averaging
        count.slice(2, pos[0], pos[0] + effective_z)
             .slice(3, pos[1], pos[1] + effective_y)
             .slice(4, pos[2], pos[2] + effective_x)
             .add_(1);
    }

    // Average overlapping regions
    result = result.div(count);

    return result;
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


torch::Tensor MonaiDataLoader::postProcessMask(torch::Tensor& segmentation)
{
    // Connected component analysis to remove small isolated regions
    // Here's a simple implementation - in practice you might want to use a library

    // First, make a binary copy of the segmentation for each class
    auto segSize = segmentation.sizes().vec();
    int numClasses = modelParams_.outputClasses;

    // Apply a 3x3x3 median filter to remove noise
    torch::Tensor filtered = segmentation.clone();

    // In a full implementation, you would add code for:
    // 1. Connected component analysis
    // 2. Size filtering (remove components smaller than threshold)
    // 3. Hole filling

    // For now, we'll just return the original segmentation
    return filtered;
}


torch::Tensor MonaiDataLoader::extractLabel(torch::Tensor& segmentation, int labelClass)
{
    // Extract a single label (e.g., pancreas) from the segmentation
    // Typically, background=0, pancreas=1, tumor=2

    // Create binary mask for the requested label
    torch::Tensor mask = (segmentation == labelClass).to(torch::kFloat32);

    return mask;
}

std::vector<int> MonaiDataLoader::computeBoundingBox(const torch::Tensor& mask)
{
    // Find the bounding box of the mask (min/max coordinates)
    at::Tensor indices = at::nonzero(mask);

    if (indices.size(0) == 0) {
        // No foreground voxels found
        return { 0, 0, 0, 0, 0, 0 };
    }

    // Get min and max coordinates for each dimension
    auto min_result = (at::min)(indices, 0);
    auto max_result = (at::max)(indices, 0);

    at::Tensor min_coords = std::get<0>(min_result);
    at::Tensor max_coords = std::get<0>(max_result);

    // Convert to vector [min_z, min_y, min_x, max_z, max_y, max_x]
    std::vector<int> bbox;
    for (int i = 0; i < min_coords.size(0); i++) {
        bbox.push_back(min_coords[i].item<int>());
    }
    for (int i = 0; i < max_coords.size(0); i++) {
        bbox.push_back(max_coords[i].item<int>());
    }

    return bbox;
}


std::map<std::string, float> MonaiDataLoader::computeVolumeStats(const torch::Tensor& mask)
{
    // Compute volume statistics 
    std::map<std::string, float> stats;

    // Count voxels
    int64_t voxelCount = torch::sum(mask).item<int64_t>();

    // Estimate volume in cubic mm (assuming you know voxel spacing)
    // For now, use a placeholder spacing of 1mm isotropic
    float spacing_x = 1.0f;
    float spacing_y = 1.0f;
    float spacing_z = 1.0f;

    float volume_mm3 = voxelCount * spacing_x * spacing_y * spacing_z;
    float volume_ml = volume_mm3 / 1000.0f;

    // Store statistics
    stats["voxel_count"] = static_cast<float>(voxelCount);
    stats["volume_mm3"] = volume_mm3;
    stats["volume_ml"] = volume_ml;

    return stats;
}

void MonaiDataLoader::saveSegmentation(const torch::Tensor& segmentation,
    const std::filesystem::path& outputPath)
{
    // Save segmentation to file (e.g., as a NIfTI file)
    // This requires a NIfTI writing library, which is not included here

    // For now, just save as a raw binary file
    std::ofstream outfile(outputPath.string(), std::ios::binary);

    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << outputPath.string() << std::endl;
        return;
    }

    // Get contiguous tensor and write to file
    torch::Tensor contiguousTensor = segmentation.contiguous().to(torch::kUInt8);
    outfile.write(static_cast<const char*>(contiguousTensor.data_ptr()),
        contiguousTensor.numel() * sizeof(uint8_t));

    outfile.close();
    std::cout << "Segmentation saved to: " << outputPath.string() << std::endl;
}