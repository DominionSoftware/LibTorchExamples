


#include <cstdlib>
#include <filesystem>
#include <memory>

#include "../Common/MONAIDataLoader.h"
#include "../Common/FileSaver.h"


using namespace torch_explorer;

int main(int argc,const char* argv[])
{
    try
    {



        auto model_data_folder = []()->const std::filesystem::path
            {
                return std::filesystem::current_path() / "RelWithDebInfo" / "pathology_nuclei_segmentation_classification" / "";

            };

        auto data_folder = []()->const std::filesystem::path
            {
                return std::filesystem::path("D:/Projects/Pancreas-CT/manifest-1599750808610/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667") / "";
                //return std::filesystem::path("D:/Projects/Images/LeftToRight.dcm") / "";

            };


#ifdef TESTDATA
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

#endif

        auto model_data = model_data_folder();

        std::shared_ptr<MonaiDataLoader> loaderSegment = std::make_shared<MonaiDataLoader>(model_data, false);



        auto data_path = data_folder();

#ifdef TESTDATA
        auto test_folders = test_folder();

        for (auto& p : test_folders)
        {
            auto t = loaderSegment->loadDicomStudy(p);
        }
#endif

        auto output_folder = []()->const std::filesystem::path
            {
                return std::filesystem::current_path() / "Output" / "";
            };

        // Create output directory if it doesn't exist
        std::filesystem::create_directories(output_folder());

        MonaiDataLoader dataLoader(model_data, false);

        // Load the DICOM study and preprocess for model
        std::cout << "Loading DICOM study from: " << data_path.string() << std::endl;

        dataLoader.loadDicomStudy(data_path);



#ifdef FUTURE

        // Print volume information
        std::cout << "Loaded volume shape: [";
        for (auto i = 0; i < volume.dim(); i++) {
            std::cout << volume.size(i) << (i < volume.dim() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Run inference to generate segmentation
        std::cout << "Running segmentation inference..." << std::endl;
        torch::Tensor segmentation = dataLoader.inference(volume);

        // Print segmentation shape
        std::cout << "Segmentation shape: [";
        for (auto i = 0; i < segmentation.dim(); i++) {
            std::cout << segmentation.size(i) << (i < segmentation.dim() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        // Extract the pancreas (assuming pancreas is label 1)
        std::cout << "Extracting pancreas mask..." << std::endl;
        torch::Tensor pancreasMask = dataLoader.extractLabel(segmentation, 1);

        // Optional: Apply post-processing to clean up the segmentation
        std::cout << "Post-processing segmentation..." << std::endl;
        pancreasMask = dataLoader.postProcessMask(pancreasMask);

        // Compute bounding box for the pancreas
        std::cout << "Computing pancreas bounding box..." << std::endl;
        std::vector<int> bbox = dataLoader.computeBoundingBox(pancreasMask);

        std::cout << "Pancreas bounding box: "
            << "[" << bbox[0] << ":" << bbox[3] << ", "
            << bbox[1] << ":" << bbox[4] << ", "
            << bbox[2] << ":" << bbox[5] << "]" << std::endl;

        // Calculate volume statistics
        std::cout << "Computing volume statistics..." << std::endl;
        auto stats = dataLoader.computeVolumeStats(pancreasMask);

        std::cout << "Pancreas volume statistics:" << std::endl;
        std::cout << "  Voxel count: " << static_cast<int>(stats["voxel_count"]) << std::endl;
        std::cout << "  Volume (mm³): " << std::fixed << std::setprecision(2) << stats["volume_mm3"] << std::endl;
        std::cout << "  Volume (ml): " << std::fixed << std::setprecision(2) << stats["volume_ml"] << std::endl;

        // Save the segmentation to a file
        auto output_path = output_folder() / "pancreas_segmentation.raw";
        std::cout << "Saving segmentation to: " << output_path.string() << std::endl;
        dataLoader.saveSegmentation(pancreasMask, output_path);

        // Save full segmentation with all labels
        auto full_output_path = output_folder() / "full_segmentation.raw";
        dataLoader.saveSegmentation(segmentation, full_output_path);

        std::cout << "Segmentation complete!" << std::endl;
#endif
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
