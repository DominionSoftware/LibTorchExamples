#ifndef MONAI_DATALOADER_
#define MONAI_DATALOADER_
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <torch/torch.h>

namespace torch_explorer
{

    class MonaiDataLoader
    {
    public:
        MonaiDataLoader(const std::filesystem::path& bundlePath, bool trainMode = true);
       
        nlohmann::json getMetadata() const;
        
        std::vector<std::pair<std::string, std::string>> loadDataPairs();

        torch::Tensor loadImage(const std::string& path, bool isLabel = false);





    private:
        torch::Tensor loadDicomImage(const std::string& path, bool isLabel = false);
        std::vector<std::pair<std::string, std::string>> loadFromConfig();
        std::vector<std::pair<std::string, std::string>> scanForImageLabelPairs();
        std::vector<std::pair<std::string, std::string>>
        scanDirectoryForImageLabelPairs(const std::filesystem::path& directory);


        std::filesystem::path bundlePath_;
        nlohmann::json config_;
        bool trainMode_;

 
        std::unordered_map<std::string, std::string> seriesDirectories_;

    };

}

#endif