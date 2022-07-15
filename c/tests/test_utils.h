// Copyright 2018 Sean Robertson
/**

Boilerplate code used in c++ tests

*/

#pragma once

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include <cmath>
#include <random>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <stack>

template <class T> int GenerateRandom(unsigned num_elements,
                                      T* out, unsigned seed = 0) {
  std::mt19937 gen(seed);
  T fan = static_cast<T>(1.0 / std::sqrt(static_cast<double>(num_elements)));
  std::uniform_real_distribution<T> dis(-fan, fan);
  for (unsigned n = 0; n < num_elements; ++n) {
    out[n] = dis(gen);
  }
  return 0;
}

int ParseNumber(const char str[], int* out) {
  char *pEnd;
  *out = std::strtol(str, &pEnd, 10);
  if (pEnd == str) {
    std::cerr << "Could not parse argument as number: " << str << std::endl;
    return 1;
  }
  return 0;
}

int ParseBoolean(const char str[], bool* out) {
  if (!std::strcmp(str, "t") || !std::strcmp(str, "T")) {
    *out = true;
  } else if (!std::strcmp(str, "f") || !std::strcmp(str, "F")) {
    *out = false;
  } else {
    std::cerr << "Argument must be 't' or 'f': " << str << std::endl;
    return 1;
  }
  return 0;
}

int ParseArgs(int argc, const char * argv[],
              int * test_idx = nullptr, int * batch = nullptr,
              bool * transposed_f = nullptr,
              bool * transposed_g = nullptr,
              bool * transposed_h = nullptr,
              int * extra_h = nullptr) {
  if ((argc < 2) || (argc > 7)) {
    std::cerr << "Invalid number of arguments: " << argc << std::endl;
    return 1;
  }
  if (test_idx && ParseNumber(argv[1], test_idx)) return 1;
  if (argc > 2 && batch && ParseNumber(argv[2], batch)) return 1;
  if (argc > 3 && transposed_f && ParseBoolean(argv[3], transposed_f)) {
    return 1;
  }
  if (argc > 4 && transposed_g && ParseBoolean(argv[4], transposed_g)) {
    return 1;
  }
  if (argc > 5 && transposed_h && ParseBoolean(argv[5], transposed_h)) {
    return 1;
  }
  if (argc > 6 && extra_h && ParseNumber(argv[6], extra_h)) return 1;
  return 0;
}

int ParseArgsRandom(int argc, const char* argv[],
                    int* c_out, int* c_in, int* batch,
                    int* nfx, int* ngx, int* nhx,
                    int* sx, int* dx, int* px, int* ux,
                    bool* transposed_f, bool* transposed_g, bool* transposed_h,
                    int* nfy, int* ngy, int* nhy,
                    int* sy, int* dy, int* py,
                    int* uy, bool* linear) {
  if ((argc != 14) || (argc != 22))
  {
    std::cerr << "Invalid number of arguments: " << argc << std::endl;
    return 1;
  }

  if (ParseNumber(argv[1], c_out)) return 1;
  if (ParseNumber(argv[2], c_in)) return 1;
  if (ParseNumber(argv[3], batch)) return 1;
  if (ParseNumber(argv[4], nfx)) return 1;
  if (ParseNumber(argv[5], ngx)) return 1;
  if (ParseNumber(argv[6], nhx)) return 1;
  if (ParseNumber(argv[7], sx)) return 1;
  if (ParseNumber(argv[8], dx)) return 1;
  if (ParseNumber(argv[9], px)) return 1;
  if (ParseNumber(argv[10], ux)) return 1;
  if (ParseBoolean(argv[11], transposed_f)) return 1;
  if (ParseBoolean(argv[12], transposed_g)) return 1;
  if (ParseBoolean(argv[13], transposed_h)) return 1;
  
  if (argc == 22) {
    if (ParseNumber(argv[14], nfy)) return 1;
    if (ParseNumber(argv[15], ngy)) return 1;
    if (ParseNumber(argv[16], nhy)) return 1;
    if (ParseNumber(argv[17], sy)) return 1;
    if (ParseNumber(argv[18], dy)) return 1;
    if (ParseNumber(argv[19], py)) return 1;
    if (ParseNumber(argv[20], uy)) return 1;
    if (ParseBoolean(argv[21], linear)) return 1;
  }
  return 0;
}

/// returns 0 on success, 1 on failure
template <class T>
int AllClose(const T* expt, const T* act, const std::vector<int>& shape,
             T eps) {
  int ret = 0;
  if (!shape.size()) return 0;
  std::vector<int> strides {1};
  for ( int elem : shape ) {
    strides.push_back(strides.back() * elem);
  }
  for (int i = 0; i < strides.back(); ++i) {
    if ((act[i] > expt[i] + eps) || (act[i] < expt[i] - eps)) {
      int rem = i;
      std::cerr << "Expected value @ ";
      for (unsigned int shape_idx = 0; shape_idx < shape.size(); ++shape_idx) {
        // std::cout << rem << " " << shape_idx << " " << shape[shape_idx] << " " << strides[shape.size() - shape_idx] << ",";
        if (shape[shape_idx] == 1) {
          std::cerr << "0,";
        } else {
          int cur = rem * strides[shape_idx + 1] / strides.back();
          rem = rem % (strides.back() / strides[shape_idx + 1]);
          std::cerr << cur << ",";
        }
      }
      std::cerr << " (" << expt[i] << ") does not match actual ("
                << act[i] << ")" << std::endl;
      ret = 1;
    }
  }
  return ret;
}

#endif  // TEST_UTILS_HPP_
