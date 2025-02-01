


#include <cstdlib>

#include "TrainModel.h"
#include <filesystem>

#include "CIFAR100DataSet.h"


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

	//std::shared_ptr<FileSaver> fileSaver = std::make_shared<FileSaver>(image_folder());


	auto dataSetTrain = std::make_shared<CIFAR100DataSet>(true);

 	dataSetTrain->load(data_folder());

	auto dataSetTest = std::make_shared<CIFAR100DataSet>(false);

	dataSetTest->load(data_folder());

	TrainModel(dataSetTrain, dataSetTest, 100);

 

	return EXIT_SUCCESS;
}
