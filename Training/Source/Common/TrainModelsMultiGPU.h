#ifndef TRAINMODELS_MULTIGPU_
#define TRAINMODELS_MULTIGPU_

#include <memory>
namespace torch_explorer
{
    class CIFAR100CoarseModule;
    class CIFAR100FineModule;
    class IDataSet;

    void TrainSplitModelsMultiGPU(
        std::shared_ptr<CIFAR100CoarseModule> coarse_model,
        std::shared_ptr<CIFAR100FineModule> fine_model,
        std::shared_ptr<IDataSet> trainData,
        std::shared_ptr<IDataSet> testData,
        size_t num_epochs,
        double coarse_lr,
        double fine_lr,
        size_t logInterval);
}
#endif