// Copyright 2021 Sean Robertson

#pragma once

#ifndef TEST_COL2SND_BUFFERS_H_
#define TEST_COL2SND_BUFFERS_H_

namespace col2sndb {

// for compatibility with the various testing suites, g remains g, but gg
// is stored in h.
// These are derived from snd2col by starting with their h and multiplying
// the values in their g by the number of times they show up in their h

const int kNumTests = 7;

const double kG[][100] = {
  {1., 4., 3., 8.},  // 0: 2f 4g
  {1., 4.}, // 1: f4 g2
  {0., 2., 3., 8., 5., 18., 7., 24., 18., 20.}, // 2: f4 g10 s=2
  {0., 2., 3., 8., 0., 12., 0.}, // 3: f3 g7 d=2
  {2., 6., 9., 12., 15., 24., 14.}, // 4: f5 g7 p=1
  {2., 4., 9., 8., 20., 6., 21., 16.}, // 5: f6 g8 u=2
  {1., 6., 3., 8., 10., 6., 21., 8., 18., 20.}, // 6: f4 g2,5 s=2, d=3, p=4, u=5
};

const int kNF[] = {
  2, 4, 4, 3, 5, 6, 4
};

const int kNG[] = {
  4, 2, 10, 7, 7, 8, 5
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
  0, 0, 0, 0, 1, 0, 4
};

const int kU[] = {
  1, 1, 1, 1, 1, 2, 5
};

const double kH[][200] = {
  {
    1., 2., 3., 4.,
    2., 4., 0., 0.
  },  // 0: f2 g4
  {
    1., 2.,
    2., 0.,
    0., 0.,
    0., 0.
  },  // 1: f4 g2
  {
    2., 3., 4., 5., 6., 7., 8., 9., 10.,
    4., 6., 8., 10., 0., 0., 0., 0., 0.,
    6., 9., 0., 0., 0., 0., 0., 0., 0.,
    8., 0., 0., 0., 0., 0., 0., 0., 0.
  },  // 2: f4 g10 s=2
  {
    2., 4., 6.,
    3., 6., 0.,
    4., 0., 0.
  },  // 3: f3 g7 d=2
  {
    0., 1., 0., 2., 0., 3., 0., 4., 0., 5., 0., 6., 0., 7.,
    1., 2., 3., 4., 5., 6., 7., 0., 0., 0., 0., 0., 0., 0.,
    0., 3., 0., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    2., 4., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
  },  // 4: f5 g7 p=1
  {
    0., 1., 2., 3., 4., 5., 6., 7., 8.,
    1., 3., 5., 7., 0., 0., 0., 0., 0.,
    2., 5., 8., 0., 0., 0., 0., 0., 0.,
    3., 7., 0., 0., 0., 0., 0., 0., 0.,
    4., 0., 0., 0., 0., 0., 0., 0., 0.,
    5., 0., 0., 0., 0., 0., 0., 0., 0.
  },  // 5: f6 g8 u=2
  {
    0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 5.,
    0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0., 0., 0., 0.,
    0., 0., 0., 1., 2., 3., 4., 5., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,

    0., 0., 0., 0., 0., 0., 0., 0., 7., 0., 0., 0., 0., 10.,
    0., 0., 0., 0., 0., 0., 0., 0., 9., 0., 0., 0., 0., 0.,
    0., 0., 0., 6., 7., 8., 9., 10., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 7., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
  },  // 6: f4 g2,5 s=2, d=3, p=4, u=5
};

}  // namespace col2sndb

#endif  // TEST_COL2SND_BUFFERS_H_