


#include <cstdlib>

#include "TrainModel.h"
#include <filesystem>

#include "CIFAR100DataSet.h"
#include "CIFAR100Module.h"
#include "CIFAR100CoarseModule.h"
#include "CIFAR100FineModule.h"
#include "TrainModelsMultiGPU.h"
#include "TrainSplitModels.h"


int main(int ,const char * args[])
{


	auto data_folder = []()->const std::filesystem::path
		{
			return std::filesystem::current_path() / "RelWithDebInfo" / "cifar-100-binary" / "";

		};

	auto image_folder = []()->const std::filesystem::path
		{
			return std::filesystem::current_path() / "RelWithDebInfo" / "images" / "";

		};

//	std::shared_ptr<torch_explorer::FileSaver> fileSaver = std::make_shared<torch_explorer::FileSaver>(image_folder());


	auto dataSetTrain = std::make_shared<torch_explorer::CIFAR100DataSet>(true);
	dataSetTrain->enableCutMix(1.0, 0.5);

 	dataSetTrain->load(data_folder());

	auto dataSetTest = std::make_shared<torch_explorer::CIFAR100DataSet>(false);

	dataSetTest->load(data_folder());


#ifdef HEIRARCHICAL
	auto model = std::make_shared<torch_explorer::CIFAR100Module>(dataSetTrain->getInputShape(),dataSetTrain->getNumClasses());


	torch_explorer::TrainModel(model, dataSetTrain, dataSetTest, 100);

#else
	auto coarseModel = std::make_shared<torch_explorer::CIFAR100CoarseModule>(dataSetTrain->getInputShape());

	auto fineModel = std::make_shared<torch_explorer::CIFAR100FineModule>(dataSetTrain->getInputShape());

	const auto device_count = torch::cuda::device_count();
	if (device_count > 1)
	{
		torch_explorer::TrainSplitModelsMultiGPU(coarseModel, fineModel, dataSetTrain, dataSetTest, 100, 0.001, 0.001,10);

	}
	else
	{
		torch_explorer::TrainSplitModels(coarseModel, fineModel, dataSetTrain, dataSetTest, 100, 0.001, 0.001);
	}

#endif
 

	return EXIT_SUCCESS;
}
