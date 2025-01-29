#ifndef CIFAR100_DATASET_
#define CIRAF100_DATASET_

#include "IDataSet.h"
#include <torch/torch.h>


class CIFAR100DataSet : public IDataSet
{
public:



private:


    torch::data::datasets::CIFAR100 dataset;
    torch::data::DataLoaderOptions options;
    std::vector<std::function<torch::data::Example<>(torch::data::Example<>)>> transforms;
    bool is_train;


};

#endif
