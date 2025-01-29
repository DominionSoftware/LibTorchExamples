#ifndef IDATA_SET_
#define IDATA_SET_

#include <torch/torch.h>

 
#include <vector>
#include <string>

#include "CIFAR100.h"


class IDataSet
{
public:
    // Virtual destructor ensures proper cleanup of derived classes
    virtual ~IDataSet() = default;

    // Load dataset from the specified directory
    virtual void load(const std::string& root_path) = 0;

    // Get a batch of data with specified indexes
    virtual torch::data::Example<> get(size_t index) = 0;

    // Get the total size of the dataset
    virtual torch::optional<size_t> size() const = 0;

    // Get batch size configured for the dataset
    virtual size_t getBatchSize() const = 0;

    // Set batch size for data loading
    virtual void setBatchSize(size_t batch_size) = 0;

    // Get number of worker threads for data loading
    virtual size_t getNumWorkers() const = 0;

    // Set number of worker threads for data loading
    virtual void setNumWorkers(size_t num_workers) = 0;

    // Check if the dataset is for training
    virtual bool isTraining() const = 0;

    // Get the image dimensions [channels, height, width]
    virtual std::vector<int64_t> getInputShape() const = 0;

    // Get number of classes in the dataset
    virtual size_t getNumClasses() const = 0;

    // Get the data loader for the dataset
    virtual auto getDataLoader() -> std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
        CIFAR100,
        torch::data::transforms::Normalize<>
        >,
        torch::data::samplers::RandomSampler
        >> = 0;

    IDataSet(){}

    IDataSet(const IDataSet&) = delete;
    IDataSet(IDataSet&&) = delete;
    IDataSet& operator=(const IDataSet&) = delete;
    IDataSet& operator=(IDataSet&&) = delete;
};

#endif