#ifndef PANCREAS_CT_DATASET_
#define PANCREAS_CT_DATASET_
#include "../Common/IDataSet.h"
#include <nlohmann/json.hpp>
namespace torch_explorer
{
    struct PancreasCT
    {
        enum class ClassType
        {
            Background = 0,
            Pancreas = 1,
        };

    };


    class PancreasCTDataset : public IDataSet<PancreasCT>

    {
    public:


    private:

        std::filesystem::path dataPath_;
        nlohmann::json dataConfig_;
        nlohmann::json datasetConfig_;

        // Data storage
        std::vector<torch::Tensor> images_;
        std::vector<torch::Tensor> labels_;

        // Dataset parameters
        std::vector<int64_t> inputShape_;
        size_t numClasses_;
        size_t batchSize_;
        bool trainMode_;

        // Cache for series directories
        std::unordered_map<std::string, std::string> seriesDirectories_;
    };


}


#endif