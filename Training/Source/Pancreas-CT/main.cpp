


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
			//return std::filesystem::path("D:/Projects/Pancreas-CT/manifest-1599750808610/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667") / "";
			return std::filesystem::path("D:/Projects/Images/BackToFront.dcm") / "";

		};

	auto test_folder = []()->const std::vector<std::filesystem::path>
		{

			std::vector<std::filesystem::path> paths =
			{
				std::filesystem::path("D:/Projects/Images/BackToFront.dcm") / "",
		        std::filesystem::path("D:/Projects/Images/FrontToBack.dcm") / "",

				std::filesystem::path("D:/Projects/Images/BottomToTop.dcm") / "",
			    std::filesystem::path("D:/Projects/Images/TopToBottom.dcm") / "",

		        std::filesystem::path("D:/Projects/Images/LeftToRight.dcm") / "",
			    std::filesystem::path("D:/Projects/Images/RightToLeft.dcm") / "",

			};

			return paths;
 
		};



	auto model_data = model_data_folder();

	std::shared_ptr< MonaiDataLoader> loaderSegment = std::make_shared<MonaiDataLoader>(model_data, false);


	loaderSegment->loadBaseModel();
	auto data_path = data_folder();
	auto test_folders = test_folder();

	for (auto& p : test_folders)
	{
		loaderSegment->loadDicomStudy(p);
	}

    return EXIT_SUCCESS;

}
