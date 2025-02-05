#ifndef CIFAR100_DATASET_
#define CIFAR100_DATASET_

#include <random>

#include "IDataSet.h"
#include <torch/torch.h>

#include "CutMixTransform.h"

namespace torch_explorer
{


    class CIFAR100DataSet : public IDataSet
    {
    public:


        CIFAR100DataSet();


        explicit CIFAR100DataSet(bool is_training);
       

       void load(const std::filesystem::path& root_path, std::shared_ptr<FileSaver> fileSaver = nullptr) override;
       

        torch::data::Example<> get(size_t index) override
        {
            return dataset.get(index);
        }

        torch::optional<size_t> size() const override
        {
            return dataset.size();
        }

        size_t getBatchSize() const override
        {
            return options.batch_size();
        }

        void setBatchSize(size_t batch_size) override
        {
            options.batch_size(batch_size);
        }

        size_t getNumWorkers() const override
        {
            return options.workers();
        }

        void setNumWorkers(size_t num_workers) override
        {
            options.workers(num_workers);
        }

        bool isTraining() const override
        {
            return is_train;
        }

        std::vector<int64_t> getInputShape() const override
        {
            return { 3, 32, 32 };  // CIFAR images are 32x32 RGB
        }

        size_t getNumClasses() const override
        {
            return 100;  // CIFAR-100 has 100 classes
        }

        auto getDataLoader() -> std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<
                                                        CIFAR100,
                                                        torch::data::transforms::Normalize<>>,
                                                        torch::data::samplers::RandomSampler>> override;
       
        void enableCutMix(float alpha = 1.0, float prob = 0.5)
        {
            use_cutmix_ = true;
            cutmix_ = CutMixTransform(alpha, prob);
        }

        void disableCutMix()
        {
            use_cutmix_ = false;
        }
    private:

        CIFAR100 dataset;

        torch::data::DataLoaderOptions options;

        std::vector<std::function<torch::data::Example<>(torch::data::Example<>)>> transforms;

        bool is_train;

        bool use_cutmix_ = false;
        CutMixTransform cutmix_{ 1.0, 0.5 };
        std::mt19937 gen_{ std::random_device{}() };
    };
}

#endif
