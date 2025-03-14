 
#ifndef MONAI_DATALOADER_
#define MONAI_DATALOADER_

#include <filesystem>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>

namespace torch_explorer {

    struct PreprocessingParams {
        float a_min = -87.0f;    // Default values from the pancreas CT model
        float a_max = 199.0f;
        float b_min = 0.0f;
        float b_max = 1.0f;
        bool clip = true;
    };

    struct ModelParams {
        int inputChannels = 1;
        int outputClasses = 3;
        std::vector<int> patchSize = { 96, 96, 96 };
        std::vector<float> inputRange = { 0.0f, 1.0f };
        float overlapRatio = 0.625f;
    };

    class MonaiDataLoader {
    public:
        MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode = false);

        // Model loading
        torch::jit::Module loadBaseModel();

        // DICOM handling
        std::vector<torch::Tensor> loadDicomStudy(const std::string& folderPath);
        torch::Tensor loadDicomVolume(const std::vector<std::string>& dicomFiles);
        void sortDicomFilesByPosition(std::vector<std::string>& dicomFiles);

        // Feature extraction for transfer learning
        torch::Tensor extractFeatures(torch::jit::Module& model, torch::Tensor& volume);
        std::vector<torch::Tensor> extractPatches(const torch::Tensor& volume, const std::vector<int>& patchSize);

        // Classifier handling for transfer learning
        torch::nn::Sequential createClassifier(int numFeatures, int numClasses);
        void saveClassifier(const torch::nn::Sequential& classifier, const std::string& path);
        torch::nn::Sequential loadClassifier(const std::string& path, int numFeatures, int numClasses);

        // Preprocessing
        torch::Tensor preprocessVolume(torch::Tensor volume);
        torch::Tensor applyIntensityScaling(const torch::Tensor& tensor);

    private:
        // Configuration loading
        void loadMetadata();
        void loadPreprocessingConfig();

        // Member variables
        std::filesystem::path bundlePath_;
        bool trainMode_;
        nlohmann::json metadata_;
        PreprocessingParams preprocessing_;
        ModelParams modelParams_;
    };

}

#endif