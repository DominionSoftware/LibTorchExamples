


#include <cstdlib>

#include "TrainModel.h"
#include <filesystem>

#include "CIFAR100DataSet.h"


int main(int ,const char * args[])
{


	auto data_folder = []()->const std::filesystem::path
		{
			return std::filesystem::current_path() / "cifar-100-binary";

		};


	auto dataSetTrain = std::make_shared<CIFAR100DataSet>(true);

	std::string fullpath = data_folder().string();
	dataSetTrain->load(fullpath);

	auto dataSetTest = std::make_shared<CIFAR100DataSet>(true);

	dataSetTest->load(data_folder().string());

	TrainModel(dataSetTrain, dataSetTest, 25);

 

	return EXIT_SUCCESS;
}
