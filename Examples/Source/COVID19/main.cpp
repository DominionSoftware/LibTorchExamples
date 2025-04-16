#include <cstdlib>

#include <filesystem>
#include "COVID19DataSet.h"
#include "Covid19Module.h"
#include "TrainCovid19Module.h"


int main(int argc, char* argv[])
{
	try
	{
		auto data_folder = []()->const std::filesystem::path
			{
				return std::filesystem::current_path() / "RelWithDebInfo" / "COVID-19_Radiography_Dataset" / "";

			};


		auto model_folder = []()->const std::filesystem::path
		
			{
				return std::filesystem::current_path() / "RelWithDebInfo" / "";

			}; 
		

		auto dataSetTrain = std::make_shared<torch_explorer::Covid19DataSet>(true);


		dataSetTrain->load(data_folder());

		auto dataSetTest = std::make_shared<torch_explorer::Covid19DataSet>(false);

		dataSetTest->load(data_folder());

		auto model_path = model_folder();

		model_path.replace_filename("resnet18_model.pt");

		
		auto coarseModel = std::make_shared<torch_explorer::Covid19Module>(model_path.string());


		torch_explorer::TrainCovid19Module(coarseModel, dataSetTrain, dataSetTest, 10);


	}
	catch (std::exception& ex)
	{
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

    return EXIT_SUCCESS;
}
