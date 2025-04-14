#ifndef DICOM_IMAGE_SLICE_
#define DICOM_IMAGE_SLICE_
#include <string>
#include <vector>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcdatset.h>

#include "DicomMetaData.h"

namespace  torch_explorer
{
    class DicomImageSlice
    {
    public:

        DicomImageSlice();
        ~DicomImageSlice();
        DicomImageSlice(const DicomImageSlice&) = delete;
        DicomImageSlice(const DicomImageSlice&&) = delete;

        DicomImageSlice& operator=(const DicomImageSlice&) = delete;
        DicomImageSlice& operator=(const DicomImageSlice&&) = delete;
        void loadImage(const DicomMetaData& metaData, const std::string& filePath, DcmFileFormat& ff);
        int getRows() const
        {
            return rows_;
        }

        int getCols() const
        {
            return cols_;
        }

        uint16_t getBitsPerPixel() const
        {
            return bitsPerPixel_;
        }

        int getIsSigned() const
        {
            return isSigned_;
        }

        std::vector<double> getImagePositionPatient() const
        {
            return imagePositionPatient_;
        }

        std::vector<double> getImageOrientationPatient() const
        {
            return imageOrientationPatient_;
        }

        int getComponentsPerPixel() const
        {
            return componentsPerPixel_;
        }

        std::vector<double> getSpacing() const
        {
            return pixelSpacing_;
        }

        size_t getPixelSizeDataInBytes() const
        {
            return pixelDataSizeInBytes_;
        }
        void* getPixelData() const
        {
            return pixelData_;
        }

    private:

        static char* allocatePixels(const DiPixel* interData, size_t& outSize);


        std::vector<double> imagePositionPatient_;
        std::vector<double> imageOrientationPatient_;
        std::vector<double> normal_;
        std::string orientationRows_;
        std::string orientationCols_;
        std::string orientationNormal_;
        double positionOnNormal_;
        double rescaleSlope_;
        double rescaleIntercept_;
        double spacingBetweenSlices_;
        double sliceThickness_;
        std::vector<double> pixelSpacing_;
        int isSigned_;
        int cols_;
        int rows_;
        uint16_t bitsPerPixel_;
        uint16_t componentsPerPixel_;
        uint16_t bitsUsedWhenWriting_;
        uint16_t highBitWhenWriting_;
        void* pixelData_;
        size_t pixelDataSizeInBytes_;
    };

   
}



#endif