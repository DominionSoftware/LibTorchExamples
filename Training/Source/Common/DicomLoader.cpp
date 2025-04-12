#include "DicomLoader.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <torch/torch.h>

#include <ATen/Functions.h>
#include "DicomImageSlice.h"
#include "DicomMetaData.h"


using namespace torch_explorer;

std::vector<DicomMetaData> DicomLoader::loadDicomStudy(const std::filesystem::path& folderPath)
{
    // Check if this is a directory containing DICOM files
    if (!is_directory(folderPath))
    {
        throw std::runtime_error("Expected a directory containing DICOM files: " + folderPath.string());
    }


    std::vector<DicomMetaData> dicomFiles;
    for (const auto& entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.is_regular_file())
        {
            DcmFileFormat fileFormat;
            if (fileFormat.loadFile(entry.path().string().c_str()).good())
            {
                DicomMetaData metaData;

                OFString studyUID;
                if (!fileFormat.getDataset()->findAndGetOFString(DCM_StudyInstanceUID, studyUID).good())
                {
                    throw std::runtime_error("Study Instance UID Not Found. " + folderPath.string());
                }

                metaData.studyUID_ = studyUID.c_str();

                OFString seriesInstanceUID;
                if (!fileFormat.getDataset()->findAndGetOFString(DCM_SeriesInstanceUID, seriesInstanceUID).good())
                {
                    throw std::runtime_error("Series Instance UID Not Found. " + folderPath.string());
                }
                {
                    metaData.seriesUID_ = seriesInstanceUID.c_str();
                }

                DcmElement* ele;
                if (!fileFormat.getDataset()->findAndGetElement(DCM_ImagePositionPatient, ele).good())
                {
                    throw std::runtime_error("Image Position Patient Not Found " + folderPath.string());
                }

                OFVector<Float64> imagePositionPatient;
                auto dcmDs = dynamic_cast<DcmDecimalString*>(ele);
                if (dcmDs != nullptr)
                {
                    if (dcmDs->getFloat64Vector(imagePositionPatient).good())
                    {
                        for (size_t i = 0; i < imagePositionPatient.size(); i++)
                        {
                            metaData.imagePositionPatient_.push_back(imagePositionPatient[i]);
                        }
                    }
                }


                if (!fileFormat.getDataset()->findAndGetElement(DCM_ImageOrientationPatient, ele).good())
                {
                    throw std::runtime_error("Image Orientation Patient Not Found " + folderPath.string());
                }

                OFVector<Float64> imageOrientationPatient;
                dcmDs = dynamic_cast<DcmDecimalString*>(ele);
                if (dcmDs != nullptr)
                {
                    if (dcmDs->getFloat64Vector(imageOrientationPatient).good())
                    {
                        for (size_t i = 0; i < imageOrientationPatient.size(); i++)
                        {
                            metaData.imageOrientationPatient_.push_back(imageOrientationPatient[i]);
                        }
                    }


                    if (metaData.imageOrientationPatient_.size() != 6)
                    {
                        throw std::runtime_error("Image Orientation Patient not correct " + folderPath.string());
                    }

                    Vector3D<double> xVector;
                    xVector[0] = metaData.imageOrientationPatient_[0];
                    xVector[1] = metaData.imageOrientationPatient_[1];
                    xVector[2] = metaData.imageOrientationPatient_[2];

                    Vector3D<double> yVector;

                    yVector[0] = metaData.imageOrientationPatient_[3];
                    yVector[1] = metaData.imageOrientationPatient_[4];
                    yVector[2] = metaData.imageOrientationPatient_[5];

                    Vector3D<double> zVector = xVector.cross(yVector);

                    std::cout << xVector << std::endl;
                    std::cout << yVector << std::endl;
                    std::cout << zVector << std::endl;

                    metaData.imageOrientationPatient_.push_back(zVector[0]);
                    metaData.imageOrientationPatient_.push_back(zVector[1]);
                    metaData.imageOrientationPatient_.push_back(zVector[2]);

                    std::vector<std::string> orientation = getFullPatientOrientation(metaData.imageOrientationPatient_);
                    if (!isLPS(metaData.imageOrientationPatient_) && !isLAI(metaData.imageOrientationPatient_))
                    {
                        throw std::runtime_error("Image Orientation Patient in this image is not supported " + folderPath.string());
                    }
                }
                dicomFiles.push_back(metaData);
            }
        }
    }

    if (dicomFiles.empty())
    {
        throw std::runtime_error("No valid DICOM files found in: " + folderPath.string());
    }
    return dicomFiles;
}


std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> DicomLoader::getImageOrientation(
    const std::string& filePath)
{
    OFVector<Float64> imageOrientationPatient;

    DcmFileFormat fileFormat;
    if (!fileFormat.loadFile(filePath.c_str()).good())
    {
        throw std::runtime_error("Unable to load DICOM file: " + filePath);
    }

    DcmElement* ele;
    DcmDataset* dataSet = fileFormat.getDataset();
    if (!dataSet->findAndGetElement(DCM_ImageOrientationPatient, ele).good())
    {
        throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
    }

    auto dcmDs = dynamic_cast<DcmDecimalString*>(ele);
    if (dcmDs == nullptr)
    {
        throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
    }

    if (!dcmDs->getFloat64Vector(imageOrientationPatient).good() || imageOrientationPatient.size() != 6)
    {
        throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
    }

    Vector3D<double> rowsVector;
    rowsVector[0] = imageOrientationPatient[0];
    rowsVector[1] = imageOrientationPatient[1];
    rowsVector[2] = imageOrientationPatient[2];

    Vector3D<double> colsVector;
    colsVector[0] = imageOrientationPatient[3];
    colsVector[1] = imageOrientationPatient[4];
    colsVector[2] = imageOrientationPatient[5];

    Vector3D<double> normalVector = Vector3D<double>::cross3D(rowsVector, colsVector);

    return std::make_tuple(rowsVector, colsVector, normalVector);
}


std::string DicomLoader::getPatientOrientation(const std::string& filePath)
{
    auto [rowsVector, colsVector, normalVector] = getImageOrientation(filePath);
    return getPatientOrientation(normalVector[0], normalVector[1], normalVector[2]);
}

// From David Clunie's Java implementation.
// 0.5477 would be the square root of 1 (unit vector sum of squares) divided by 3 (oblique axes - a "double" oblique)
// 0.7071 would be the square root of 1 (unit vector sum of squares) divided by 2 (oblique axes)

std::string DicomLoader::getPatientOrientation(double x, double y, double z)
{
    /// \brief The obliquity threshold cosine value.
    constexpr double obliquityThresholdCosineValue = 0.8;
    std::string axis;

    std::string orientationX(x < 0 ? "R" : "L");
    std::string orientationY(y < 0 ? "A" : "P");
    std::string orientationZ(z < 0 ? "I" : "S");
    double absX = fabs(x);
    double absY = fabs(y);
    double absZ = fabs(z);

    // The tests here really don't need to check the other dimensions,
    // just the threshold, since the sum of the squares should be == 1.0
    // but just in case ...

    if (absX > obliquityThresholdCosineValue && absX > absY && absX > absZ)
    {
        axis = orientationX;
    }
    else if (absY > obliquityThresholdCosineValue && absY > absX && absY > absZ)
    {
        axis = orientationY;
    }
    else if (absZ > obliquityThresholdCosineValue && absZ > absX && absZ > absY)
    {
        axis = orientationZ;
    }
    return axis;
}


bool DicomLoader::isLPS(std::vector<double>& iop)
{
    assert(iop.size() == 9);

    std::vector<std::string> orientation = getFullPatientOrientation(iop);
    // x axis runs from patient “R”ight to “L”eft
    // y axis runs from patient “A”nterior to “P”osterior
    // z axis runs from patient “I”nferior to “S”uperior.
    return (orientation[0] == "L" && orientation[1] == "P" && orientation[2] == "S");
}


bool DicomLoader::isLAI(std::vector<double>& iop)
{
    assert(iop.size() == 9);

    std::vector<std::string> orientation = getFullPatientOrientation(iop);
    // x axis runs from patient “R”ight to “L”eft
    // y axis runs from patient “P”osterior to “A”nterior
    // z axis runs from patient “S”uperior to "I"nferior
    return (orientation[0] == "L" && orientation[1] == "A" && orientation[2] == "I");
}


std::vector<std::string> DicomLoader::getFullPatientOrientation(std::vector<double>& iop)
{
    std::string orientationRows = getPatientOrientation(iop[0], iop[1], iop[2]);
    std::string orientationCols = getPatientOrientation(iop[3], iop[4], iop[5]);

    std::vector<double>& cosines = iop;
    std::vector<double> normal;
    normal.resize(3);
    normal[0] = cosines[1] * cosines[5] - cosines[2] * cosines[4];
    normal[1] = cosines[2] * cosines[3] - cosines[0] * cosines[5];
    normal[2] = cosines[0] * cosines[4] - cosines[1] * cosines[3];
    std::string orientationNormal = getPatientOrientation(normal[0], normal[1], normal[2]);

    std::vector<std::string> result = {orientationRows, orientationCols, orientationNormal};
    return result;
}


torch::Tensor DicomLoader::loadDicomVolume(const std::vector<std::string>& dicomFiles)
{
    // Parameters for the 3D volume
    size_t width = 0;
    size_t height = 0;
    size_t depth = dicomFiles.size();
    std::vector<float> pixelData;

    // Read each slice
    for (const auto& filePath : dicomFiles)
    {
    }


    return torch::Tensor();
}
