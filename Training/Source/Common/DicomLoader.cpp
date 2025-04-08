#include "DicomLoader.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <torch/torch.h>
 
#include <ATen/Functions.h>

namespace torch_explorer
{
    torch::Tensor DicomLoader::loadDicomStudy(const std::filesystem::path& folderPath)
    {
        // Check if this is a directory containing DICOM files
        if (!is_directory(folderPath))
        {
            throw std::runtime_error("Expected a directory containing DICOM files: " + folderPath.string());
        }

        // Collect all DICOM files
        std::vector<std::string> dicomFiles;
        for (const auto& entry : std::filesystem::directory_iterator(folderPath))
        {
            if (entry.is_regular_file())
            {
                DcmFileFormat fileFormat;
                if (fileFormat.loadFile(entry.path().string().c_str()).good())
                {
                    dicomFiles.push_back(entry.path().string());
                }
            }
        }

        if (dicomFiles.empty())
        {
            throw std::runtime_error("No valid DICOM files found in: " + folderPath.string());
        }

        // Sort files by position
        std::vector<std::string> sortedFiles = sortDicomFilesByPosition(dicomFiles);
        auto imageOrientation = getImageOrientation(dicomFiles[0]);





        directionCosinesMatrix_ = createTransformFromDirectionCosines(dicomFiles[0]);

        torch::Tensor volume = loadDicomVolume(sortedFiles);

        std::vector<std::string> fullOrientation = getFullPatientOrientation(std::get<0>(imageOrientation), std::get<1>(imageOrientation), std::get<2>(imageOrientation));
        std::cout << fullOrientation << std::endl;

        {
            volume = applyAffineTransform(volume,directionCosinesMatrix_);
        }
        return loadDicomVolume(sortedFiles);
    }
#pragma optimize("",off)

    std::tuple<Vector3D<double>, Vector3D<double>, Vector3D<double>> DicomLoader::getImageOrientation(const std::string& filePath)
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

    Vector3D<double> DicomLoader::getImagePosition(const std::string& filePath)
    {
        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(filePath.c_str()).good())
        {
            throw std::runtime_error("Unable to load DICOM file: " + filePath);
        }

        DcmElement* ele;
        DcmDataset* dataSet = fileFormat.getDataset();
        if (!dataSet->findAndGetElement(DCM_ImagePositionPatient, ele).good())
        {
            throw std::runtime_error("Image Position Patient not valid in file: " + filePath);
        }

        OFVector<Float64> imagePositionPatient;
        auto dcmDs = dynamic_cast<DcmDecimalString*>(ele);
        if (dcmDs == nullptr || !dcmDs->getFloat64Vector(imagePositionPatient).good() || imagePositionPatient.size() !=
            3)
        {
            throw std::runtime_error("Image Position Patient not valid in file: " + filePath);
        }

        Vector3D<double> position;
        position[0] = imagePositionPatient[0];
        position[1] = imagePositionPatient[1];
        position[2] = imagePositionPatient[2];

        return position;
    }

    std::string DicomLoader::getDicomMetadata(const std::string& filePath, const DcmTagKey& tagKey)
    {
        DcmFileFormat fileFormat;
        if (!fileFormat.loadFile(filePath.c_str()).good())
        {
            throw std::runtime_error("Unable to load DICOM file: " + filePath);
        }

        OFString value;
        if (fileFormat.getDataset()->findAndGetOFString(tagKey, value).good())
        {
            return std::string(value.c_str());
        }

        return "";
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

        if (absX > obliquityThresholdCosineValue && absX > absY && absX > absZ) {
            axis = orientationX;
        }
        else if (absY > obliquityThresholdCosineValue && absY > absX && absY > absZ) {
            axis = orientationY;
        }
        else if (absZ > obliquityThresholdCosineValue && absZ > absX && absZ > absY) {
            axis = orientationZ;
        }
        return axis;
    }
 


    std::vector<std::string> DicomLoader::getFullPatientOrientation(Vector3D<double> rows, Vector3D<double> cols, Vector3D<double> normal)
    {
        std::string orientationRows = getPatientOrientation(rows[0], rows[1], rows[2]);
        std::string orientationCols = getPatientOrientation(cols[0], cols[1], cols[2]);

        std::string orientationNormal = getPatientOrientation(normal[0], normal[1], normal[2]);

        std::vector<std::string> result = { orientationRows, orientationCols, orientationNormal };
        return result;
    }

    std::vector<std::string> DicomLoader::sortDicomFilesByPosition(const std::vector<std::string>& dicomFiles)
    {
        if (dicomFiles.empty())
        {
            return {};
        }

        // Get orientation from the first file
        auto [rowsVector, colsVector, normalVector] = getImageOrientation(dicomFiles[0]);
        std::string patientOrientation = DicomLoader::getPatientOrientation(normalVector[0], normalVector[1], normalVector[2]);

        std::cout << patientOrientation << std::endl;
        
        std::vector<std::pair<std::string, double>> filePositions;

        for (const auto& filePath : dicomFiles)
        {
            try
            {
                Vector3D<double> position = getImagePosition(filePath);
                if (patientOrientation == "P" || patientOrientation == "A")
                {
                    // sort along y;
                    filePositions.emplace_back(filePath, position[1]);
                    std::cout << "sorting along Y " << position[1] << std::endl;

                }
                else if (patientOrientation == "S" || patientOrientation == "I")
                {
                    filePositions.emplace_back( filePath, position[2] );
                    std::cout << "sorting along Z " << position[2] << std::endl;

                }
                else
                {
                    filePositions.emplace_back(filePath, position[0]);
                    std::cout << "sorting along X " << position[0] << std::endl;

                }

            }
            catch (const std::exception& e)
            {
                std::cerr << "Warning: " << e.what() << std::endl;
            }
        }

        // Sort by position
        std::sort(filePositions.begin(), filePositions.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Extract only the file paths
        std::vector<std::string> sortedFiles;
        sortedFiles.reserve(filePositions.size());

        for (const auto& [filePath, d] : filePositions)
        {
            std::cout << d << std::endl;
            sortedFiles.push_back(filePath);
        }

        return sortedFiles;
    }



    Eigen::Matrix4d DicomLoader::createTransformFromDirectionCosines(const std::string& dicomFilePath)
    {
        // Get the orientation vectors
        auto [rowsVector, colsVector, normalVector] = getImageOrientation(dicomFilePath);

        // Create 4x4 transform matrix
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

        // Set rotation part from direction cosines
        transform(0, 0) = rowsVector[0];
        transform(0, 1) = rowsVector[1];
        transform(0, 2) = rowsVector[2];

        transform(1, 0) = colsVector[0];
        transform(1, 1) = colsVector[1];
        transform(1, 2) = colsVector[2];

        transform(2, 0) = normalVector[0];
        transform(2, 1) = normalVector[1];
        transform(2, 2) = normalVector[2];

        // Note: We're not setting the translation part here
        // If needed, you can incorporate that from image position

        return transform;
    }

    torch::Tensor DicomLoader::applyAffineTransform(const torch::Tensor& volume,const Eigen::Matrix4d& transform)
    {
        // Get input volume dimensions
        auto sizes = volume.sizes().vec();
        int batch_size = sizes[0];
        int channels = sizes[1];
        int depth = sizes[2];
        int height = sizes[3];
        int width = sizes[4];

        // Create output tensor with the same dimensions
        torch::Tensor output = torch::zeros_like(volume);

        // Get inverse transform for backward mapping
        Eigen::Matrix4d inv_transform = transform.inverse();

        // Process each batch and channel separately
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                // Center coordinates for the transform
                float center_z = (depth - 1) / 2.0f;
                float center_y = (height - 1) / 2.0f;
                float center_x = (width - 1) / 2.0f;

                // Loop through each voxel in the output
                for (int z = 0; z < depth; z++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            // Convert to centered coordinates
                            float cx = x - center_x;
                            float cy = y - center_y;
                            float cz = z - center_z;

                            // Apply inverse transform to get source coordinates
                            float src_x = inv_transform(0, 0) * cx + inv_transform(0, 1) * cy +
                                inv_transform(0, 2) * cz + inv_transform(0, 3);
                            float src_y = inv_transform(1, 0) * cx + inv_transform(1, 1) * cy +
                                inv_transform(1, 2) * cz + inv_transform(1, 3);
                            float src_z = inv_transform(2, 0) * cx + inv_transform(2, 1) * cy +
                                inv_transform(2, 2) * cz + inv_transform(2, 3);

                            // Convert back to volume coordinates
                            src_x += center_x;
                            src_y += center_y;
                            src_z += center_z;

                            // Check if coordinates are within bounds
                            if (src_x >= 0 && src_x < width - 1 &&
                                src_y >= 0 && src_y < height - 1 &&
                                src_z >= 0 && src_z < depth - 1) {

                                // Trilinear interpolation
                                int x0 = std::floor(src_x);
                                int y0 = std::floor(src_y);
                                int z0 = std::floor(src_z);

                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                // Clamp to valid range
                                x0 = std::max<int>(0, std::min<int>(width - 1, x0));
                                y0 = std::max<int>(0, std::min<int>(height - 1, y0));
                                z0 = std::max<int>(0, std::min<int>(depth - 1, z0));

                                x1 = std::max<int>(0, std::min<int>(width - 1, x1));
                                y1 = std::max<int>(0, std::min<int>(height - 1, y1));
                                z1 = std::max<int>(0, std::min<int>(depth - 1, z1));

                                // Interpolation weights
                                float wx = src_x - x0;
                                float wy = src_y - y0;
                                float wz = src_z - z0;

                                // Get values at corner points
                                float v000 = volume[b][c][z0][y0][x0].item<float>();
                                float v001 = volume[b][c][z0][y0][x1].item<float>();
                                float v010 = volume[b][c][z0][y1][x0].item<float>();
                                float v011 = volume[b][c][z0][y1][x1].item<float>();
                                float v100 = volume[b][c][z1][y0][x0].item<float>();
                                float v101 = volume[b][c][z1][y0][x1].item<float>();
                                float v110 = volume[b][c][z1][y1][x0].item<float>();
                                float v111 = volume[b][c][z1][y1][x1].item<float>();

                                // Interpolate
                                float v00 = v000 * (1 - wx) + v001 * wx;
                                float v01 = v010 * (1 - wx) + v011 * wx;
                                float v10 = v100 * (1 - wx) + v101 * wx;
                                float v11 = v110 * (1 - wx) + v111 * wx;

                                float v0 = v00 * (1 - wy) + v01 * wy;
                                float v1 = v10 * (1 - wy) + v11 * wy;

                                float value = v0 * (1 - wz) + v1 * wz;

                                // Set the output value
                                output[b][c][z][y][x] = value;
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

#pragma optimize("",off)


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
            DcmFileFormat fileFormat;
            OFCondition status = fileFormat.loadFile(filePath.c_str());

            if (status.good())
            {
                DicomImage image(&fileFormat, EXS_Unknown);

                if (image.getStatus() == EIS_Normal)
                {
                    // If first slice, get dimensions
                    if (width == 0)
                    {
                        width = image.getWidth();
                        height = image.getHeight();
                        pixelData.reserve(width * height * depth);
                    }

                    // Access raw pixel data
                    const DiPixel* pixelMatrix = image.getInterData();
                    if (pixelMatrix)
                    {
                        auto data = const_cast<void*>(pixelMatrix->getData());
                        if (data)
                        {
                            // Get technical parameters
                            uint16_t bitsAllocated = 0;
                            if (fileFormat.getDataset()->findAndGetUint16(DCM_BitsAllocated, bitsAllocated).good())
                            {


                                // For image data, convert to Hounsfield Units if CT
                                double rescaleIntercept = 0.0;
                                double rescaleSlope = 1.0;
                                fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, rescaleIntercept);
                                fileFormat.getDataset()->findAndGetFloat64(DCM_RescaleSlope, rescaleSlope);

                                // Process pixel data based on bit depth
                                if (bitsAllocated == 16)
                                {
                                    // Get pixel representation (signed or unsigned)
                                    uint16_t pixelRepresentation = 0;
                                    if (fileFormat.getDataset()->findAndGetUint16(DCM_PixelRepresentation, pixelRepresentation).good())
                                    {
                                        if (pixelRepresentation == 0)
                                        {
                                            // Unsigned data
                                            auto pixelValues = static_cast<const uint16_t*>(data);
                                            for (unsigned long i = 0; i < width * height; i++)
                                            {
                                                float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope +
                                                    rescaleIntercept;
                                                pixelData.push_back(hounsfield);
                                            }
                                        }
                                        else
                                        {
                                            // Signed data
                                            auto pixelValues = static_cast<const int16_t*>(data);
                                            for (unsigned long i = 0; i < width * height; i++)
                                            {
                                                float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope + rescaleIntercept;
                                                pixelData.push_back(hounsfield);
                                            }
                                        }
                                    }
                                }
                                else if (bitsAllocated == 8)
                                {
                                    // 8-bit data
                                    auto pixelValues = static_cast<const uint8_t*>(data);
                                    for (unsigned long i = 0; i < width * height; i++)
                                    {
                                        float hounsfield = static_cast<float>(pixelValues[i]) * rescaleSlope +
                                            rescaleIntercept;
                                        pixelData.push_back(hounsfield);
                                    }
                                }
                                else
                                {
                                    throw std::runtime_error("Unsupported bit depth: " + std::to_string(bitsAllocated));
                                }
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
