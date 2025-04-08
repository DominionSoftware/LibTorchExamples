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
#include <eigen/Eigen>

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

        /**
        * @brief Apply an affine transformation to a volume tensor
        *
        * @param volume Input volume tensor of shape [B, C, D, H, W]
        * @param transform 4x4 affine transformation matrix
        * @return Transformed volume tensor
        */
        torch::Tensor applyAffineTransform(const torch::Tensor& volume, const Eigen::Matrix4d& transform);


        static Eigen::Matrix4d createTransformFromDirectionCosines(const std::string& filePath);

        static std::string getPatientOrientation(const std::string& filePath);

        static std::string getPatientOrientation(double x, double y, double z);

        static std::vector<std::string> getFullPatientOrientation(Vector3D<double> rows, Vector3D<double> cols, Vector3D<double> normal);

        static std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> getImageOrientation(const std::string& filePath);

        std::string patientOrientation_;

        Eigen::Matrix4d directionCosinesMatrix_;

    };

} 

#endif