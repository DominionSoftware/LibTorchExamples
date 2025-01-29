#ifndef CIFAR100_DATASET_
#define CIFAR100_DATASET_

#include "IDataSet.h"
#include <torch/torch.h>
 

class CIFAR100DataSet : public IDataSet
{
public:


    CIFAR100DataSet() :IDataSet(), is_train(false)
    {
        options = torch::data::DataLoaderOptions()
            .batch_size(32)
            .workers(2);
    }
    

    explicit CIFAR100DataSet(bool is_training) : IDataSet(),is_train(is_training)
    {
        options = torch::data::DataLoaderOptions()
            .batch_size(32)
            .workers(2);
    }

    void load(const std::string& root_path) override
    {
        dataset = CIFAR100(
            root_path,
            is_train ? CIFAR100::Mode::kTrain
            : CIFAR100::Mode::kTest
        );
    }

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

    auto getDataLoader() -> std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
        CIFAR100,
        torch::data::transforms::Normalize<>
        >,
        torch::data::samplers::RandomSampler
        >> override
    {
        auto normalized_dataset = dataset
            .map(torch::data::transforms::Normalize<>({ 0.5071, 0.4867, 0.4408 },
                { 0.2675, 0.2565, 0.2761 }));

        return torch::data::make_data_loader(std::move(normalized_dataset), options);
    }

private:
    CIFAR100 dataset;
    torch::data::DataLoaderOptions options;
    std::vector<std::function<torch::data::Example<>(torch::data::Example<>)>> transforms;
    bool is_train;
};

#endif
