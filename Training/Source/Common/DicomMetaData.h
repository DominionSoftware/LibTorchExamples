#ifndef DICOM_META_DATA_
#define DICOM_META_DATA_
#include <filesystem>


namespace torch_explorer
{
    struct DicomMetaData
    {
        std::filesystem::path filePath_;
        std::string studyUID_;
        std::string seriesUID_;
        std::vector<double> imagePositionPatient_;
        std::vector<double> imageOrientationPatient_;
        std::vector<double> pixelSpacing_;

        double rescaleSlope_;
        double rescaleIntercept_;
    };

}

#endif


