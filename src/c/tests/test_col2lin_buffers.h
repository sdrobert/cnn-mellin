// Copyright 2021 Sean Robertson

#pragma once

#ifndef TEST_COL2LIN_BUFFERS_H_
#define TEST_COL2LIN_BUFFERS_H_

namespace col2linb {

// for compatibility with the various testing suites, g remains g, but gg
// is stored in h.
// These are derived from lin2col by starting with their h and multiplying
// the values in their g by the number of times they show up in their h

const int kNumTests = 7;

const double kG[][100] = {
  {1., 4., 9., 16., 25., 30., 35.},  // 0: f5 g7
  {1., 4., 9., 16., 25.}, // 1: f7 g5
  {1., 2., 6., 8., 10., 12., 14., 16.}, // 2: f4 g8 s=2
  {1., 2., 6., 8., 15., 18., 21., 24., 27.}, // 3: f3 g9 d=2
  {4., 10., 18., 24., 30., 36.}, // 4: f6 g6 p=3
  {1., 6., 15., 20.}, // 5: f5 g4 u=2
  {1., 2., 6., 4., 10., 6., 7., 16., 9., 20.}, // 6: f3 g2,5 s=2, d=3, p=4, u=5
};

const int kNF[] = {
  5, 7, 4, 3, 6, 5, 3
};

const int kNG[] = {
  7, 5, 8, 9, 6, 4, 5
};

const int kCIn[] = {
  1, 1, 1, 1, 1, 1, 2
};

const int kS[] = {
  1, 1, 2, 1, 1, 1, 2
};

const int kD[] = {
  1, 1, 1, 2, 1, 1, 3
};

const int kP[] = {
  0, 0, 0, 0, 3, 0, 4
};

const int kU[] = {
  1, 1, 1, 1, 1, 2, 5
};

const double kH[][200] = {
  {
    1., 2., 3., 4., 5., 6., 7.,
    2., 3., 4., 5., 6., 7., 0.,
    3., 4., 5., 6., 7., 0., 0.,
    4., 5., 6., 7., 0., 0., 0.,
    5., 6., 7., 0., 0., 0., 0.
  },  // 0: f5 g7
  {
    1., 2., 3., 4., 5.,
    2., 3., 4., 5., 0.,
    3., 4., 5., 0., 0.,
    4., 5., 0., 0., 0.,
    5., 0., 0., 0., 0.,
    0., 0., 0., 0., 0.,
    0., 0., 0., 0., 0.
  }, // 1: f7 g5
  {
    1., 3., 5., 7.,
    2., 4., 6., 8.,
    3., 5., 7., 0.,
    4., 6., 8., 0.
  }, // 2: f4 g8 s=2
  {
    1., 2., 3., 4., 5., 6., 7., 8., 9.,
    3., 4., 5., 6., 7., 8., 9., 0., 0.,
    5., 6., 7., 8., 9., 0., 0., 0., 0.
  }, // 3: f3 g9 d=2
  {
    0., 0., 0., 1., 2., 3., 4., 5., 6.,
    0., 0., 1., 2., 3., 4., 5., 6., 0.,
    0., 1., 2., 3., 4., 5., 6., 0., 0.,
    1., 2., 3., 4., 5., 6., 0., 0., 0.,
    2., 3., 4., 5., 6., 0., 0., 0., 0.,
    3., 4., 5., 6., 0., 0., 0., 0., 0.
  }, // 4: f6 g6 p=3
  {
    1., 0., 2., 0., 3., 0., 4.,
    0., 2., 0., 3., 0., 4., 0.,
    2., 0., 3., 0., 4., 0., 0.,
    0., 3., 0., 4., 0., 0., 0.,
    3., 0., 4., 0., 0., 0., 0.
  }, // 5: f5 g4 u=2
  {
    0., 0., 1., 0., 0., 0., 0., 3., 0., 0., 0., 0., 5.,
    0., 0., 0., 2., 0., 0., 0., 0., 4., 0., 0., 0., 0.,
    0., 0., 0., 0., 3., 0., 0., 0., 0., 5., 0., 0., 0.,

    0., 0., 6., 0., 0., 0., 0., 8., 0., 0., 0., 0., 10.,
    0., 0., 0., 7., 0., 0., 0., 0., 9., 0., 0., 0., 0.,
    0., 0., 0., 0., 8., 0., 0., 0., 0., 10., 0., 0., 0.
  }, // 6: f3 g2,5 s=2, d=3, p=4, u=5
};

}  // namespace col2linb

#endif  // TEST_COL2LIN_BUFFERS_H_