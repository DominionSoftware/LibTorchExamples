#ifndef MONAI_DATALOADER_
#define MONAI_DATALOADER_

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "Vector3D.h"
#include "DicomLoader.h"
#include "JSONLoader.h"

namespace torch_explorer {

    class MonaiDataLoader {
    public:
        // Structures to hold configuration parameters
        struct PreprocessingParams {
            float a_min = 0.0f;
            float a_max = 0.0f;
            float b_min = 0.0f;
            float b_max = 1.0f;
            bool clip = true;
        };

        struct ModelParameters {
            std::vector<int> patchSize = { 96, 96, 96 };  // Default patch size
            float overlapRatio = 0.5f;                    // Default overlap ratio
            int inputChannels = 1;                        // Default input channels
            int outputClasses = 3;                        // Default output classes (background, pancreas, tumor)
            std::array<float, 2> inputRange = { 0.0f, 1.0f }; // Default input range
            int swBatchSize = 4;                          // Default sliding window batch size
            std::string paddingMode = "constant";         // Default padding mode
            float paddingValue = 0.0f;                    // Default padding value
            bool showProgress = true;                     // Default progress indicator
            std::vector<int> extraPadding = { 0, 0, 0 };  // Default extra padding
        };

        MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode = false);

        // Load the base MONAI DiNTS model
        torch::jit::Module loadBaseModel();

        // Load a DICOM study and preprocess it for the model
        std::vector<torch::Tensor> loadDicomStudy(const std::filesystem::path& folderPath);

        // Extract features from a volume using the model
        torch::Tensor extractFeatures(torch::jit::Module& model, torch::Tensor& volume);

        // Configuration loading methods
        void loadMetadata();
        void loadPreprocessingConfig();

        // Getter methods for model parameters
        const ModelParameters& getModelParams() const { return modelParams_; }
        const PreprocessingParams& getPreprocessingParams() const { return preprocessing_; }

    private:
        // Preprocessing methods
        torch::Tensor preprocessVolume(torch::Tensor volume);
        torch::Tensor applyIntensityScaling(const torch::Tensor& tensor);
        std::vector<torch::Tensor> extractPatches(const torch::Tensor& volume, const std::vector<int>& patchSize);

        // Parse model parameters from configuration
        void parseModelParameters(nlohmann::json& config);
        void parsePreprocessingParameters(nlohmann::json& config);

        // Member variables
        std::filesystem::path bundlePath_;
        bool trainMode_;
        nlohmann::json metadata_;
        PreprocessingParams preprocessing_;
        ModelParameters modelParams_;

        // Helper classes
        DicomLoader dicomLoader_;
        JSONLoader jsonLoader_;
    };

}

#endif