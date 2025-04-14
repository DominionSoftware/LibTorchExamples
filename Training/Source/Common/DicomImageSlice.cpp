#include "DicomImageSlice.h"

#include <dcmtk/ofstd/oftypes.h>

#include "DicomMetaData.h"

using namespace torch_explorer;


DicomImageSlice::DicomImageSlice() : rescaleSlope_(1.0), rescaleIntercept_(0.0), positionOnNormal_(0.0), spacingBetweenSlices_(1),
                                sliceThickness_(1), isSigned_(0), cols_(0), rows_(0), bitsPerPixel_(0),
                                componentsPerPixel_(0), pixelData_(nullptr), pixelDataSizeInBytes_(0)
{
    
}


DicomImageSlice::~DicomImageSlice()
{
    if (pixelData_ != nullptr)
    {
        free(pixelData_);
    }
}

void DicomImageSlice::loadImage(const DicomMetaData& metaData,const std::string& filePath,DcmFileFormat& ff)
{
    Uint16 bitsAllocated;
    Uint16 bitsStored;
    Uint16 highBit;
    Uint16 isSigned;

    if (!ff.getDataset()->findAndGetUint16(DCM_BitsAllocated, bitsAllocated).good())
    {
        throw std::runtime_error("Unable to load image.");
    }

    if (!ff.getDataset()->findAndGetUint16(DCM_BitsStored, bitsStored).good())
    {
        throw std::runtime_error("Unable to load image.");
    }

    if (!ff.getDataset()->findAndGetUint16(DCM_HighBit, highBit).good())
    {
        throw std::runtime_error("Unable to load image.");
    }

    if (!ff.getDataset()->findAndGetUint16(DCM_PixelRepresentation, isSigned).good())
    {
        throw std::runtime_error("Unable to load image.");
    }

    Uint16 rows;

    if (!ff.getDataset()->findAndGetUint16(DCM_Rows, rows).good())
    {
        throw std::runtime_error("Unable to load image.");
    }
    Uint16 cols;
    if (!ff.getDataset()->findAndGetUint16(DCM_Columns, cols).good())
    {
        throw std::runtime_error("Unable to load image.");
    }

    double intercept;
    if (!ff.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, intercept).good())
    {
        intercept = 0.0;
    }
    double slope;
    if (!ff.getDataset()->findAndGetFloat64(DCM_RescaleSlope, slope).good())
    {
        slope = 1.0;
    }

    Uint16 samplesPerPixel;
    if (!ff.getDataset()->findAndGetUint16(DCM_SamplesPerPixel, samplesPerPixel).good())
    {
        throw std::runtime_error("Unable to load image.");
    }


    DicomImage image(filePath.c_str(), 0);
    const DiPixel* interData = image.getInterData();


    auto status = image.getStatus();
    if (status != EIS_Normal)
    {
        throw std::runtime_error("Bad image data.");
    }



    size_t sizeInBytes = 0;
    char* buffer = allocatePixels(interData, sizeInBytes);
    if (buffer == nullptr)
    {
        throw std::runtime_error("Out of memory.");
    }
    EP_Representation rep = interData->getRepresentation();


    switch (rep)
    {
    case EPR_Uint8:
        bitsPerPixel_ = 8;
        componentsPerPixel_ = 1;
        isSigned_ = false;
        break;
    case EPR_Sint8:
        bitsPerPixel_ = 8;
        componentsPerPixel_ = 1;
        isSigned_ = true;
        break;
    case EPR_Uint16:
        bitsPerPixel_ = 16;
        componentsPerPixel_ = 1;
        isSigned_ = false;
        break;
    case EPR_Sint16:
        bitsPerPixel_ = 16;
        componentsPerPixel_ = 1;
        isSigned_ = true;
        break;
    case EPR_Uint32:
        bitsPerPixel_ = 32;
        componentsPerPixel_ = 1;
        isSigned_ = false;
        break;
    case EPR_Sint32:
        bitsPerPixel_ = 32;
        componentsPerPixel_ = 1;
        isSigned_ = true;
        break;
    }

    imagePositionPatient_ = metaData.imagePositionPatient_;
    imageOrientationPatient_ = metaData.imageOrientationPatient_;
    pixelSpacing_ = metaData.pixelSpacing_;
    cols_ = cols;
    rows_ = rows;
    pixelData_ = buffer;
    pixelDataSizeInBytes_ = sizeInBytes;
    rescaleIntercept_ = metaData.rescaleIntercept_;
    rescaleSlope_ = metaData.rescaleSlope_;
}



char* DicomImageSlice::allocatePixels(const DiPixel* const interData, size_t& outSize)
{
    size_t componentSize = 0;
    uint32_t componentCount = interData->getInputCount();
    int numPlanes = interData->getPlanes();
    EP_Representation rep = interData->getRepresentation();
    switch (rep)
    {
    case EPR_Uint8:
    case EPR_Sint8:
        componentSize = 1;
        break;

    case EPR_Uint16:
    case EPR_Sint16:
        componentSize = 2;
        break;
    case EPR_Uint32:
    case EPR_Sint32:
        componentSize = 4;
        break;
    default:  // NOLINT(clang-diagnostic-covered-switch-default)
        throw std::runtime_error("Bad Component Size.");
        break;
    }

    outSize = (componentSize * numPlanes) * componentCount;

    return static_cast<char*>(malloc(outSize));
}