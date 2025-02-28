
#include "CIFAR100DataSet.h"
#include "CIFAR100ClassNames.h"

using namespace torch_explorer;



CIFAR100DataSet::CIFAR100DataSet() : IDataSet(), is_train(false)
{
    options = torch::data::DataLoaderOptions()
        .batch_size(32)
        .workers(2);
}


CIFAR100DataSet::CIFAR100DataSet(bool is_training) : IDataSet(), is_train(is_training)
{
    options = torch::data::DataLoaderOptions()
        .batch_size(32)
        .workers(2);
}



void CIFAR100DataSet::load(const std::filesystem::path& root_path, std::shared_ptr<FileSaver> fileSaver)
{

    CIFAR100ClassNames::instance().loadClassNames(root_path, "coarse_label_names.txt", "fine_label_names.txt");

    dataset = torch_explorer::CIFAR100();
    ProgressBar<int64_t> bar;

    dataset.load(root_path, is_train ? CIFAR100::Mode::kTrain : CIFAR100::Mode::kTest, bar, fileSaver);
}



auto CIFAR100DataSet::getDataLoader() -> std::unique_ptr<torch::data::StatelessDataLoader<
    torch::data::datasets::MapDataset<
    CIFAR100,
    torch::data::transforms::Normalize<>
    >,
    torch::data::samplers::RandomSampler
    >>
{
    auto normalized_dataset = dataset
        .map(torch::data::transforms::Normalize<>({ 0.5071, 0.4867, 0.4408 },
            { 0.2675, 0.2565, 0.2761 }));
    return torch::data::make_data_loader(std::move(normalized_dataset), options);

}