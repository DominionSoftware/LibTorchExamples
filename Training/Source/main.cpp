


#include <cstdlib>

#include "TrainModel.h"
#include <filesystem>

#include "CIFAR100DataSet.h"
#include "CIFAR100Module.h"


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

	std::shared_ptr<torch_explorer::FileSaver> fileSaver = std::make_shared<torch_explorer::FileSaver>(image_folder());


	auto dataSetTrain = std::make_shared<torch_explorer::CIFAR100DataSet>(true);

 	dataSetTrain->load(data_folder(), fileSaver);

	auto dataSetTest = std::make_shared<torch_explorer::CIFAR100DataSet>(false);

	dataSetTest->load(data_folder());

	auto model = std::make_shared<torch_explorer::CIFAR100Module>(dataSetTrain->getInputShape(),dataSetTrain->getNumClasses());


	torch_explorer::TrainModel(model, dataSetTrain, dataSetTest, 100);

 

	return EXIT_SUCCESS;
}
