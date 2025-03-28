#include "DicomLoader.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace torch_explorer {

    torch::Tensor DicomLoader::loadDicomStudy(const std::filesystem::path& folderPath) 
    {
        // Check if this is a directory containing DICOM files
        if (!std::filesystem::is_directory(folderPath)) 
        {
            throw std::runtime_error("Expected a directory containing DICOM files: " + folderPath.string());
        }

        // Collect all DICOM files
        std::vector<std::string> dicomFiles;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) 
        {
            if (entry.is_regular_file()) {
                DcmFileFormat fileFormat;
                if (fileFormat.loadFile(entry.path().string().c_str()).good()) 
                {
                    dicomFiles.push_back(entry.path().string());
                }
            }
        }

        if (dicomFiles.empty()) {
            throw std::runtime_error("No valid DICOM files found in: " + folderPath.string());
        }

        // Sort files by position
        std::vector<std::string> sortedFiles = sortDicomFilesByPosition(dicomFiles);

        // Load the volume
        return loadDicomVolume(sortedFiles);
    }

    std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> DicomLoader::getImageOrientation(const std::string& filePath) {
        OFVector<Float64> imageOrientationPatient;

        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(filePath.c_str()).good()) {
            throw std::runtime_error("Unable to load DICOM file: " + filePath);
        }

        DcmElement* ele;
        DcmDataset* dataSet = fileFormat.getDataset();
        if (!dataSet->findAndGetElement(DCM_ImageOrientationPatient, ele).good()) {
            throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
        }

        DcmDecimalString* dcmDs = dynamic_cast<DcmDecimalString*>(ele);
        if (dcmDs == nullptr) {
            throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
        }

        if (!dcmDs->getFloat64Vector(imageOrientationPatient).good() || imageOrientationPatient.size() != 6) {
            throw std::runtime_error("Image Orientation Patient not valid in file: " + filePath);
        }

        Vector3D<double> xVector;
        xVector[0] = imageOrientationPatient[0];
        xVector[1] = imageOrientationPatient[1];
        xVector[2] = imageOrientationPatient[2];

        Vector3D<double> yVector;
        yVector[0] = imageOrientationPatient[3];
        yVector[1] = imageOrientationPatient[4];
        yVector[2] = imageOrientationPatient[5];

        Vector3D<double> zVector = Vector3D<double>::cross3D(xVector, yVector);

        return std::make_tuple(xVector, yVector, zVector);
    }

    Vector3D<double> DicomLoader::getImagePosition(const std::string& filePath) {
        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(filePath.c_str()).good()) {
            throw std::runtime_error("Unable to load DICOM file: " + filePath);
        }

        DcmElement* ele;
        DcmDataset* dataSet = fileFormat.getDataset();
        if (!dataSet->findAndGetElement(DCM_ImagePositionPatient, ele).good()) {
            throw std::runtime_error("Image Position Patient not valid in file: " + filePath);
        }

        OFVector<Float64> imagePositionPatient;
        DcmDecimalString* dcmDs = dynamic_cast<DcmDecimalString*>(ele);
        if (dcmDs == nullptr || !dcmDs->getFloat64Vector(imagePositionPatient).good() || imagePositionPatient.size() != 3) {
            throw std::runtime_error("Image Position Patient not valid in file: " + filePath);
        }

        Vector3D<double> position;
        position[0] = imagePositionPatient[0];
        position[1] = imagePositionPatient[1];
        position[2] = imagePositionPatient[2];

        return position;
    }

    std::string DicomLoader::getDicomMetadata(const std::string& filePath, const DcmTagKey& tagKey) {
        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(filePath.c_str()).good()) {
            throw std::runtime_error("Unable to load DICOM file: " + filePath);
        }

        OFString value;
        if (fileFormat.getDataset()->findAndGetOFString(tagKey, value).good()) {
            return std::string(value.c_str());
        }

        return "";
    }

    std::vector<std::string> DicomLoader::sortDicomFilesByPosition(const std::vector<std::string>& dicomFiles) {
        if (dicomFiles.empty()) {
            return {};
        }

        // Get orientation from the first file
        auto [xVector, yVector, zVector] = getImageOrientation(dicomFiles[0]);

        // Create a vector of pairs: (file path, position along Z-axis)
        std::vector<std::pair<std::string, double>> filePositions;

        for (const auto& filePath : dicomFiles) {
            try {
                Vector3D<double> position = getImagePosition(filePath);

                // Calculate position along the normal vector (Z-axis)
                double zPosition = position[0] * zVector[0] +
                    position[1] * zVector[1] +
                    position[2] * zVector[2];

                filePositions.push_back({ filePath, zPosition });
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        }

        // Sort by position
        std::sort(filePositions.begin(), filePositions.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        // Extract only the file paths
        std::vector<std::string> sortedFiles;
        sortedFiles.reserve(filePositions.size());

        for (const auto& [filePath, _] : filePositions) {
            sortedFiles.push_back(filePath);
        }

        return sortedFiles;
    }

    torch::Tensor DicomLoader::loadDicomVolume(const std::vector<std::string>& dicomFiles) {
        // Parameters for the 3D volume
        unsigned long width = 0, height = 0;
        size_t depth = dicomFiles.size();
        std::vector<float> pixelData;

        // Read each slice
        for (const auto& filePath : dicomFiles) {
            DcmFileFormat fileFormat;
            OFCondition status = fileFormat.loadFile(filePath.c_str());

            if (status.good()) {
                DicomImage image(&fileFormat, EXS_Unknown);

                if (image.getStatus() == EIS_Normal) {
                    // If first slice, get dimensions
                    if (width == 0) {
                        width = image.getWidth();
                        height = image.getHeight();
                        pixelData.reserve(width * height * depth);
                    }

                    // Access raw pixel data
                    const DiPixel* pixelMatrix = image.getInterData();
                    if (pixelMatrix) {
                        auto data = const_cast<void*>(pixelMatrix->getData());
                        if (data) {
                            // Get technical parameters
                            int bitsAllocated = 0;
                            fileFormat.getDataset()->findAndGetSint32(DCM_BitsAllocated, bitsAllocated);

                            // For image data, convert to Hounsfield Units if CT
                            double rescaleIntercept = 0.0;
                            double rescaleSlope = 1.0;
                            fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, rescaleIntercept);
                            fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleSlope, rescaleSlope);

                            // Process pixel data based on bit depth
                            if (bitsAllocated == 16) {
                                // Get pixel representation (signed or unsigned)
                                int pixelRepresentation = 0;
                                fileFormat.getDataset()->findAndGetSint32(DCM_PixelRepresentation, pixelRepresentation);

                                if (pixelRepresentation == 0) {
                                    // Unsigned data
                                    auto pixelValues = static_cast<const uint16_t*>(data);
                                    for (unsigned long i = 0; i < width * height; i++) {
                                        float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope + rescaleIntercept;
                                        pixelData.push_back(hounsfield);
                                    }
                                }
                                else {
                                    // Signed data
                                    auto pixelValues = static_cast<const int16_t*>(data);
                                    for (unsigned long i = 0; i < width * height; i++) {
                                        float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope + rescaleIntercept;
                                        pixelData.push_back(hounsfield);
                                    }
                                }
                            }
                            else if (bitsAllocated == 8) {
                                // 8-bit data
                                auto pixelValues = static_cast<const uint8_t*>(data);
                                for (unsigned long i = 0; i < width * height; i++) {
                                    float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope + rescaleIntercept;
                                    pixelData.push_back(hounsfield);
                                }
                            }
                            else {
                                throw std::runtime_error("Unsupported bit depth: " + std::to_string(bitsAllocated));
                            }
                        }
                    }
                }
            }
        }

        // Create torch tensor from the pixel data
        torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

        auto tensor = torch::from_blob(pixelData.data(),
            {
                1, // Batch dimension
                1, // Channel dimension (for MONAI models)
                static_cast<long>(depth),
                static_cast<long>(height),
                static_cast<long>(width)
            },
            options).clone();

        return tensor;
    }

} // namespace torch_explorer