#ifndef DICOM_LOADER_
#define DICOM_LOADER_

#include <string>
#include <vector>
#include <filesystem>
#include <iosfwd>
#include <tuple>
#include <vector>
#include <vtkImageData.h>
#include <torch/torch.h>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcdatset.h>
#include "Vector3D.h"
#include <eigen/Eigen>

#include "DicomMetaData.h"

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
        std::vector<DicomMetaData> loadDicomStudy(const std::filesystem::path& folderPath);
        vtkSmartPointer<vtkImageData> loadToVTK(const std::vector<DicomMetaData>& metaData, double majorAxisSpacing);

        static bool isLPS(std::vector<double>& iop);

    private:
       



        static std::string getPatientOrientation(const std::string& filePath);

        static std::string getPatientOrientation(double x, double y, double z);
       
        static bool isLAI(std::vector<double>& iop);

        static std::vector<std::string> getFullPatientOrientation(std::vector<double>& iop);
        torch::Tensor loadDicomVolume(const std::vector<std::string>& dicomFiles);

        static std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> getImageOrientation(const std::string& filePath);

        std::string patientOrientation_;

        Eigen::Matrix4d directionCosinesMatrix_;

    };

} 

#endif