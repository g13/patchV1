#include "visualStim.h"

int main()
{
#ifdef FREEIMAGE_LIB
    FreeImage_Initialise();
#endif 
    FreeImage_SetOutputMessage(ErrorHandler);
    string imgName = "download";
    string ext = ".jpg";
    string imgFileName = imgName + ext;
    cout << "loading " << imgFileName.c_str() << "\n";
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(imgFileName.c_str(),0);
    FIBITMAP* img;
    if (format != FIF_UNKNOWN) {
	    img = FreeImage_Load(format, "download.jpg");
    } else {
        cout << "image format unrecognized" << "\n";
        return 0;
    }
    BYTE *bits;
    if (!FreeImage_HasPixels(img)) {
        cout << "image contains no pixel data" << "\n";
        return 0;   
    } else {
        bits = (BYTE *)FreeImage_GetBits(img);
    }
    unsigned width = FreeImage_GetWidth(img);
    unsigned height = FreeImage_GetHeight(img);
    unsigned pitch = FreeImage_GetPitch(img);
    cout << "Figure size in pixel: " << width << " x " << height << "\n";
    FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(img);
    unsigned bpp = FreeImage_GetBPP(img);
    cout << "pixel in " << bpp << " bits" << "\n";
    FREE_IMAGE_TYPE imgType = FreeImage_GetImageType(img);
    if (imgType == FIT_UNKNOWN) {
        cout << "img data type unrecognized" << "\n";
        return 0;
    } else if (imgType == FIT_COMPLEX) {
        cout << "complex pixel type not supported" << "\n";
        return 1;
    } 

    vector<vector<float>> Lcone(height,vector<float>(width,0));
    vector<vector<float>> Mcone(height,vector<float>(width,0));
    vector<vector<float>> Scone(height,vector<float>(width,0));
    vector<vector<float>> Rod(height,vector<float>(width,0));
    vector<vector<float>> channelR(height,vector<float>(width,0));
    vector<vector<float>> channelG(height,vector<float>(width,0));
    vector<vector<float>> channelB(height,vector<float>(width,0));

    if (imgType == FIT_UINT16)  {
        getBWi<uint16_t>(channelR, channelG, channelB, bits, width, height, pitch, bpp);
    } else if (imgType == FIT_INT16) {
        getBWi<int16_t>(channelR, channelG, channelB, bits, width, height, pitch, bpp);
    } else if (imgType == FIT_UINT32) {
        getBWi<uint32_t>(channelR, channelG, channelB, bits, width, height, pitch, bpp);
    } else if (imgType == FIT_INT32) {
        getBWi<int32_t>(channelR, channelG, channelB, bits, width, height, pitch, bpp);
    } else if (imgType == FIT_FLOAT) {
        getBWf<float>(channelR, channelG, channelB, bits, width, height, pitch);
    } else if (imgType == FIT_DOUBLE) {
        cout << "pixel of double size will be compressed to float" << "\n"; 
        getBWf<double>(channelR, channelG, channelB, bits, width, height, pitch);
    } else if (imgType == FIT_BITMAP) {
        if (colorType == FIC_MINISBLACK || colorType == FIC_MINISWHITE) {
            if (bpp < 8) {
                cout << "1-bit image and 4-bit image not supported" << "\n";
                return 1;
            } else {
                getBWi<unsigned char>(channelR, channelG, channelB, bits, width, height, pitch, bpp, colorType);
            }
        } else if (colorType == FIC_RGB) {
            if (bpp == 24 || bpp == 32) {
                unsigned nc = pow(2,8); // number of color representable
                unsigned denorm = nc - 1;
                for(unsigned i=0; i<height; i++) {
                    BYTE *pixel = (BYTE*) bits;
                    for(unsigned j=0; j<width; j++) {
                        channelR[i][j] = static_cast<float>(pixel[FI_RGBA_RED])/denorm;
                        channelG[i][j] = static_cast<float>(pixel[FI_RGBA_GREEN])/denorm;
                        channelB[i][j] = static_cast<float>(pixel[FI_RGBA_BLUE])/denorm;
                        pixel += bpp/8;
                    }
                    bits += pitch;
                }
            } else {
                assert(bpp == 16);
                unsigned redMask = FreeImage_GetRedMask(img);
                unsigned greenMask = FreeImage_GetGreenMask(img);
                unsigned blueMask = FreeImage_GetBlueMask(img);
                unsigned redShift, greenShift, blueShift;
                unsigned nc = pow(2,5);
                unsigned denorm = nc - 1;
                unsigned ncG, denormG;
                if (FI16_565_RED_MASK == redMask && FI16_565_GREEN_MASK == greenMask) {
                    ncG = pow(2,6);
                    redShift = FI16_565_RED_SHIFT;
                    greenShift = FI16_565_GREEN_SHIFT;
                    blueShift = FI16_565_BLUE_SHIFT;
                } else {
                    ncG = nc;
                    redShift = FI16_555_RED_SHIFT;
                    greenShift = FI16_555_GREEN_SHIFT;
                    blueShift = FI16_555_BLUE_SHIFT;
                }
                denormG = ncG - 1;
                for(unsigned i=0; i<height; i++) {
                    unsigned short *pixel = (unsigned short *) bits;
                    for(unsigned j=0; j<width; j++) {
                        channelR[i][j] = static_cast<float>((*pixel & redMask) >> redShift)/denorm;
                        channelG[i][j] = static_cast<float>((*pixel & greenMask) >> greenShift)/denormG;
                        channelB[i][j] = static_cast<float>((*pixel & blueMask) >> blueShift)/denorm;
                        pixel += bpp/8;
                    }
                    bits += pitch;
                }
            }
        } else if (colorType == FIC_PALETTE) {
            cout << "Palette image not supported" << "\n";
            return 1;
        } else if (colorType == FIC_CMYK) {
            cout << "CMYK space not supported" << "\n";
            return 1;
        }
    } else if (imgType == FIT_RGBF) {
        for (unsigned i=0; i<height; i++) {
            FIRGBF *pixel = (FIRGBF *) bits;
            for (unsigned j=0; j<width; j++) {
                channelR[i][j] = pixel[j].red;
                channelG[i][j] = pixel[j].green;
                channelB[i][j] = pixel[j].blue;
            }
            bits += pitch;
        }
    } else if (imgType == FIT_RGBAF) {
        for (unsigned i=0; i<height; i++) {
            FIRGBAF *pixel = (FIRGBAF *) bits;
            for (unsigned j=0; j<width; j++) {
                channelR[i][j] = pixel[j].red;
                channelG[i][j] = pixel[j].green;
                channelB[i][j] = pixel[j].blue;
            }
            bits += pitch;
        }
    } else if (imgType == FIT_RGB16) {
        for (unsigned i=0; i<height; i++) {
            FIRGB16 *pixel = (FIRGB16 *) bits;
            for (unsigned j=0; j<width; j++) {
                channelR[i][j] = pixel[j].red;
                channelG[i][j] = pixel[j].green;
                channelB[i][j] = pixel[j].blue;
            }
            bits += pitch;
        }
    } else if (imgType == FIT_RGBA16) {
        for (unsigned i=0; i<height; i++) {
            FIRGBA16 *pixel = (FIRGBA16 *) bits;
            for (unsigned j=0; j<width; j++) {
                channelR[i][j] = pixel[j].red;
                channelG[i][j] = pixel[j].green;
                channelB[i][j] = pixel[j].blue;
            }
            bits += pitch;
        }
    } 
    // Unload source image
    FreeImage_Unload(img);
    cout << "unloaded" << "\n";

    // Verify
    string matchImageName = imgName + "_matching";
    string outputExt = ".tiff";
    FREE_IMAGE_FORMAT outputFormat = FIF_TIFF;
    FREE_IMAGE_TYPE outputType = FIT_RGBF;
    FIBITMAP *matchImg = FreeImage_AllocateT(outputType,width,height,3*sizeof(float));
    cout << "output bpp: " << 3*sizeof(float) << "\n";
    if (matchImg==NULL) {
        cout << "allocate new image failed" << "\n";
        return 1;
    } else {
        cout << "allocation success" << "\n";
    }
    BYTE *byte = (BYTE *)FreeImage_GetBits(matchImg);
    pitch = FreeImage_GetPitch(matchImg); // scan width in bytes
    cout << "width " << width << "\n";
    cout << "scan-width: " << pitch << "\n";
    for (unsigned i=0; i<height; i++) {
        FIRGBF *pixel = (FIRGBF *)byte;
        for (unsigned j=0; j<width; j++) {
            pixel[j].red = channelR[i][j];
            pixel[j].green = channelG[i][j];
            pixel[j].blue = channelB[i][j];
        }
        cout << " row " << i << "filled\n";
        byte += pitch;
    }
    if(FreeImage_FIFSupportsWriting(outputFormat) && FreeImage_FIFSupportsExportBPP(outputFormat, bpp)) {
        if (FreeImage_Save(outputFormat, matchImg, (matchImageName + outputExt).c_str(),0)) {
            cout << "image saved" << "\n";
        } else {
            cout << "failed to save" << "\n";
        }
    } else {
        cout << "can't write " << FreeImage_GetFormatFromFIF(outputFormat) << " format" << "\n";
    }
#ifdef FREEIMAGE_LIB
    FreeImage_DeInitialise();
#endif
    return 0;
}
