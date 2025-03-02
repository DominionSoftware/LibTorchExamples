#ifndef TRAINMODELS_MULTIGPU_
#define TRAINMODELS_MULTIGPU_

#include <memory>
#include <vector>
#include "CIFAR100.h"
#include "IDataSet.h"

namespace torch_explorer
{
    class CIFAR100CoarseModule;
    class CIFAR100FineModule;
 
    void TrainSplitModelsMultiGPU(
        std::shared_ptr<CIFAR100CoarseModule> coarse_model,
        std::shared_ptr<CIFAR100FineModule> fine_model,
        std::shared_ptr<IDataSet<CIFAR100>> trainData,
        std::shared_ptr<IDataSet<CIFAR100>> testData,
        size_t num_epochs,
        double coarse_lr,
        double fine_lr,
        size_t logInterval);
}
#endif