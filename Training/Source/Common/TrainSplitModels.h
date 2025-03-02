#ifndef TRAIN_SPLIT_MODELS_
#define TRAIN_SPLIT_MODELS_

#include <memory>
#include "IDataSet.h"
#include "CIFAR100.h"



namespace torch_explorer
{
 
    
    class CIFAR100CoarseModule;
    class CIFAR100FineModule;
     void TrainSplitModels(std::shared_ptr<CIFAR100CoarseModule> coarse_model,
        std::shared_ptr<CIFAR100FineModule> fine_model,
        std::shared_ptr<IDataSet<CIFAR100>> trainData,
        std::shared_ptr<IDataSet<CIFAR100>> testData,
        size_t num_epochs,
        double coarse_lr,
        double fine_lr,
        size_t logInterval = 10);
}

#endif