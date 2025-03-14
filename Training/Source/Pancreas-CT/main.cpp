


#include <cstdlib>
#include <filesystem>
#include <memory>

 #include "../Common/MONAIDataLoader.h"



using namespace torch_explorer;

int main(int argc,const char* argv[])
{


	auto model_data_folder = []()->const std::filesystem::path
		{
			return std::filesystem::current_path() / "RelWithDebInfo" / "pathology_nuclei_segmentation_classification" / "";

		};


	auto model_data = model_data_folder();

	std::shared_ptr< MonaiDataLoader> loaderTrain = std::make_shared<MonaiDataLoader>(model_data, true);


	loaderTrain->loadBaseModel();



    return EXIT_SUCCESS;

}
