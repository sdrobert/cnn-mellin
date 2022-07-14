// Copyright 2018 Sean Robertson

#pragma once

#ifndef TEST_LCORR1D_BUFFERS_H_
#define TEST_LCORR1D_BUFFERS_H_

namespace lcorr1db {

const int kNumTests = 8;

const double kF[][100] = {
  { 4., -10., 0. },  // 0: 3f 10g (p=2)
  { 0., 4., -5., 7., 4., -7., 1., -3., 4. },  // 1: 9f 2g (p=8)
  { 1., 8. },  // 2: 2f 4g
  { -1., -2., 3. },  // 3: 3f 7g (s=3)
  { 7., -5., 2., 4. },  // 4: 4f 5g (d=4)
  { 0., 7., -1., -2. },  // 5: 4f 7g (s=2,d=3,p=1)
  { -1., 2., -1. },  // 6: 3f 4g (u=2)
  { 3., -1. },  // 7: 2f 3g (s=2,d=3,p=1,u=4)
};

const double kG[][100] = {
  { -3., -5., 9., 2., 0., -8., 0., -6., 2., -4. },  // 0: 3f 10g (p=2)
  { -7., -6. },  // 1: 9f 2g (p=8)
  { 3., 5., 7., 9. },  // 2: 2f 4g
  { 3., -3., -2., 2., 1., -1., 5. },  // 3: 3f 7g (s=3)
  { 4., 1., 6., 2., 3. },  // 4: 4f 5g (d=4)
  { 8., 1., 0., -2., 9., 0., -7. },  // 5: 4f 7g (s=2,d=3,p=1)
  { 3., 2., 1., 3. },  // 6: 3f 4g (u=2)
  { 4., 1., 4. },  // 7: 2f 3g (s=2,d=3,p=1,u=4)
};

const int kNF[] = {
  3, 9, 2, 3, 4, 4, 3, 2
};

const int kNG[] = {
  10, 2, 4, 7, 5, 7, 4, 3
};

const int kCIn[] = {
  1, 1, 1, 1, 1, 1, 1, 1
};

const int kCOut[] = {
  1, 1, 1, 1, 1, 1, 1, 1
};

const int kS[] = {
  1, 1, 1, 3, 1, 2, 1, 2
};

const int kD[] = {
  1, 1, 1, 1, 4, 3, 1, 3
};

const int kP[] = {
  2, 8, 0, 0, 0, 1, 0, 1
};

const int kU[] = {
  1, 1, 1, 1, 1, 1, 2, 4
};

const double kH[][100] = {
  {
    0., 30., 38., -110., 16., 8., 80., -32., 60., -44., 48., -16.
  },  // 0: 3f 10g (p=2)
  { -28., -3., 11., 43., 14., -73., -7., 2., -24., 0. },  // 1: 9f 2g (p=8)
  { 43., 61., 79., 9. },  // 2: 2f 4g
  { -3., -7., -5. },  // 3: 3f 7g (s=3)
  { 13., 7., 42., 14., 21. },  // 4: 4f 5g (d=4)
  { 0., 63., -49., 0. },  // 5: 4f 7g (s=2,d=3,p=1)
  { -5., 4., -3., 2., -4., 6., -3. },  // 6: 3f 4g (u=2)
  { 0., -1., 0., -4., 0. },  // 7: 2f 3g (s=2,d=3,p=1,u=4)
};

}

#endif  // TEST_LCORR1D_BUFFERS_H_
