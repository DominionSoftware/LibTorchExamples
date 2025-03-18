


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

	auto data_folder = []()->const std::filesystem::path
		{
			return std::filesystem::path("D:/Projects/Pancreas-CT/manifest-1599750808610/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667") / "";
		};

	auto model_data = model_data_folder();

	std::shared_ptr< MonaiDataLoader> loaderSegment = std::make_shared<MonaiDataLoader>(model_data, false);


	loaderSegment->loadBaseModel();
	auto data_path = data_folder();

	loaderSegment->loadDicomStudy(data_path);



    return EXIT_SUCCESS;

}
