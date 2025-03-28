#ifndef DICOM_LOADER_
#define DICOM_LOADER_

#include <string>
#include <vector>
#include <filesystem>
#include <tuple>
#include <torch/torch.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include "Vector3D.h"

namespace torch_explorer 
{

    class DicomLoader 
    {
    public:
        /**
         * @brief Loads a DICOM study from a folder containing DICOM files.
         *
         * @param folderPath Path to the folder containing DICOM files
         * @return A tensor representing the loaded volume
         */
        torch::Tensor loadDicomStudy(const std::filesystem::path& folderPath);

        /**
         * @brief Gets the image orientation from a DICOM file.
         *
         * @param filePath Path to the DICOM file
         * @return A tuple containing the X, Y, and Z orientation vectors
         */
        std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> getImageOrientation(const std::string& filePath);

        /**
         * @brief Gets the DICOM metadata from a file.
         *
         * @param filePath Path to the DICOM file
         * @param tagKey DICOM tag to retrieve (e.g., DCM_PatientName)
         * @return The value of the specified DICOM tag as a string
         */
        std::string getDicomMetadata(const std::string& filePath, const DcmTagKey& tagKey);

    private:
        /**
         * @brief Sorts DICOM files by position.
         *
         * @param dicomFiles Vector of paths to DICOM files
         * @return A sorted vector of DICOM file paths
         */
        std::vector<std::string> sortDicomFilesByPosition(const std::vector<std::string>& dicomFiles);

        /**
         * @brief Loads a 3D volume from a collection of DICOM files.
         *
         * @param dicomFiles Vector of paths to DICOM files (should be sorted)
         * @return A tensor representing the loaded volume
         */
        torch::Tensor loadDicomVolume(const std::vector<std::string>& dicomFiles);

        /**
         * @brief Gets the image position from a DICOM file.
         *
         * @param filePath Path to the DICOM file
         * @return A Vector3D containing the image position
         */
        Vector3D<double> getImagePosition(const std::string& filePath);
    };

} 

#endif