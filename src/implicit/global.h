#ifndef GLOBAL
#define GLOBAL

// texture for kernel convolution with the input
texture<float, cudaTextureType2DLayered> L_retinaConSig;
texture<float, cudaTextureType2DLayered> M_retinaConSig;
texture<float, cudaTextureType2DLayered> S_retinaConSig;

#endif
