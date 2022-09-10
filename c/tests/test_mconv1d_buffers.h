// Copyright 2018 Sean Robertson

#pragma once

#ifndef TEST_MCONV1D_BUFFERS_H_
#define TEST_MCONV1D_BUFFERS_H_

namespace mconv1db {

const int kNumTests = 13;

const double kF[][100] = {
  {  // 0: 15f 2g
    1./3., 3./7.,  -2.,    9./5., -1./9.,
    2./3., -1./5., 1./2.,  6.,    3./2.,
    -5.,   -7./5., -4./9., 5.,    3./2.
  },
  {  // 1: 8f 3g
    -7./6., 2.,  -7./4., -2./7., -10./7.,
    -5./3., -5., 9./7.
  },
  { -8./5., 3./8., 3./4., -1./3. },  // 2: 4f 4g
  { 6./7., -1./4., 2. },  // 3: 3f 10g
  { -1./2. },  // 4: 1f 2g
  { 5. },  // 5: 1f 1g
  { 1., 2., 3. },  // 6: 3f 6g (p=1)
  { 3., 2., 1. },  // 7: 3f 6g (s=2)
  { -1., 1. },  // 8: 2f 4g (d=2)
  { 10., 5., 3. },  // 9: 3f 3g (s=2,d=2,p=1)
  { 2., 1., 3., 4. },  // 10: 4f 2g (u=2)
  { -1., 1., 2. },  // 11: 3f 4g (p=1,s=3,d=4,u=2)
  { -2., 1., 4., 3., 2., 8., 9., -10. },  // 12: 8f 1g (p=3)
};

const double kG[][100] = {
  { 3., -9./7. },  // 0: 15f 2g
  { -8./9., -5./4., 5./3. },  // 1: 8f 3g
  { 1./2., 1./4., -3./5., 10./9. },  // 2: 4f 4g
  {  // 3: 3f 10g
    -4./7., -5./2., -5./7., 2./5., -9./10.,
    8./9., 10./9., -3./2., -1./9., -3./8.,
  },
  { 7./4., 7./8. },  // 4: 1f 2g
  { 6./10. },  // 5: 1f 1g
  { 6., 5., 4., 3., 2., 1. },  // 6: 3f 6g (p=1)
  { 1., 2., 3., 4., 5., 6. },  // 7: 3f 6g (s=2)
  { 1., -1., 1., -1. },  // 8: 2f 4g (d=2)
  { -2., 1., 6. },  // 9: 3f 3g (s=2,d=2,p=1)
  { 4., -1. },  // 10: 4f 2g (u=2)
  { 2., 3., 4., 1. },  // 11: 3f 4g (p=1,s=3,d=4,u=2)
  { 1. },  // 12: 8f 1g (p=3)
};

const int kNF[] = {
  15, 8, 4, 3, 1, 1, 3, 3, 2, 3, 4, 3, 8
};

const int kNG[] = {
  2, 3, 4, 10, 2, 1, 6, 6, 4, 3, 2, 4, 1
};

const int kCIn[] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

const int kCOut[] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

const int kS[] = {
  1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1
};

const int kD[] = {
  1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 4, 1
};

const int kP[] = {
  0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 3
};

const int kU[] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1
};

const double kH[][100] = {
  {
    1.,      6./7.,     -6.,       1188./245., -1./3.,
    32./7.,  -3./5.,    -57./70.,  18.,       65./14.,
    -15.,    -177./35., -4./3.,    534./35.,  9./2.,
    -9./14., 0.,        -54./7.,   0.,        -27./14.,
    0.,      45./7.,    0.,        9./5.,     0.,
    4./7.,   0.,        -45./7.,   0.,        -27./14.,
  },  // 0: 15f 2g
  {
    28./27.,    -23./72., -7./18.,  -283./126., 80./63.,
    3025./432., 40./9.,   -11./14., -35./12.,   25./14.,
    0.,         45./28,   0.,       25./4.,     -50./21.,
    -45./28.,   0.,       -25./9.,  0.,         0.,
    -25/3.,     0.,       0.,       45./21.
  },  // 1: 8f 3g
  {
    -4./5.,   -17./80., 267./200., -533./288., 0.,
    -3./80.,  0.,       1./3.,     -9./20.,    0.,
    0.,       31./30.,  0.,        0.,         0.,
    -10./27.
  },  // 2: 4f 4g
  {
    -24./49.,    -2.,     -86./49., 271./280., -27./35.,
    -1023./252., 20./21., -97./70., -32./21.,  -27./280.,
    0.,          26./45., 0.,       -5./18.,   -9./5.,
    3./8.,       0.,      65./36.,  0.,        3./32.,
    20./9.,      0.,      0.,       -9./3.,    0.,
    0.,          -2./9.,  0.,       0.,        -3./4.
  },  // 3: 3f 10g
  { -7./8., -7./16. },  // 4: 1f 2g
  { 3. },  // 5: 1f 1g
  {
    17., 13., 24., 6., 4.,
    11., 0.,  0.,  3.
  },  // 6: 3f 6g (p=1)
  {
    8.,  10., 16., 15., 26.,
    0.,  8.,  3.,  10., 0.,
    16., 0.,  0.,  5.,  0.,
    0.,  6.
  },  // 7: 3f 6g (s=2)
  {
    0.,  -1., 1., 1., 0.,
    -2., 0.,  1., 1., 0.,
    0.,  -1.
  },  // 8: 2f 4g (d=2)
  { 4., 65., 3., 0., 18. },  // 9: 3f 3g (s=2,d=2,p=1)
  {
    0.,  8.,  -2., 4.,  0.,
    11., 0.,  16., -3., 0.,
    0.,  -4.
  },  // 10: 4f 2g (u=2)
  {
    0., -2., 2., 1., 0.,
    -4., 6., 3., 0., 8.,
    0., 0., 2.
  },  // 11: 3f 4g (p=1,s=3,d=4,u=2)
  { 3., -10. }  // 12: 8f 1g (p=3)
};

}  // namespace mconv1db

#endif  // TEST_MCONV1D_BUFFERS_H_