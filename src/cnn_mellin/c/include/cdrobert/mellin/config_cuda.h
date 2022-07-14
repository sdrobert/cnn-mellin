#pragma once

#ifndef CDROBERT_MELLIN_CONFIG_CUDA_H_
#define CDROBERT_MELLIN_CONFIG_CUDA_H_

namespace cdrobert {
namespace mellin {

const int kMConv1DCudaAlgorithmVersion = 1;
const int kMCorr1DCudaAlgorithmVersion = 1;
const int kMConvLConvCudaAlgorithmVersion = 1;
const int kMCorrLCorrCudaAlgorithmVersion = 1;
const int kSnd2ColCudaAlgorithmVersion = 1;
const int kCol2SndCudaAlgorithmVersion = 1;
const int kSpec2ColCudaAlgorithmVersion = 1;
const int kCol2SpecCudaAlgorithmVersion = 1;
const bool kCudaSerial = false;
const int kMaxCudaThreadsPerBlock = 1024;

}  // mellin
}  // cdrobert

#endif  // CDROBERT_MELLIN_CONFIG_CUDA_H_
