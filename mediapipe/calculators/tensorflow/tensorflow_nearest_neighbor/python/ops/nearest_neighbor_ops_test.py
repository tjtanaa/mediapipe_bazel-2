# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for zero_out ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_nearest_neighbor.python.ops.nearest_neighbor_ops import nearest_neighbor
except ImportError:
  from nearest_neighbor_ops import nearest_neighbor


class KNNTest(test.TestCase):

  def testKNNTest(self):
    with self.test_session():
      Label = [[[0]],[[1]],[[0]],[[0]],[[2]],[[0]],[[0]]]
      X = [[[1, 2], [30, 40], [99,100]],
          [[1, 2], [3, 4], [99,100]],
          [[1, 2], [0, 40], [99,100]],
          [[1, 2], [0, 4], [99,100]],
          [[1, 2], [3, 0], [99,100]],
          [[1, 20], [30, 40], [99,100]],
          [[10, 2], [30, 40], [99,100]]]
      Query = [[[1, 2]],
              [[30, 40]],
              [[1, 2]],
              [[1, 2]],
              [[100, 99]],
              [[1, 2]],
              [[1, 2]]]
      self.assertAllClose(
          nearest_neighbor(X, Query, 1), np.array(Label))


if __name__ == '__main__':
  # print("Start")
  # X = [[[1, 2], [30, 40], [99,100]],
  #     [[1, 2], [3, 4], [99,100]],
  #     [[1, 2], [0, 40], [99,100]],
  #     [[1, 2], [0, 4], [99,100]],
  #     [[1, 2], [3, 0], [99,100]],
  #     [[1, 20], [30, 40], [99,100]],
  #     [[10, 2], [30, 40], [99,100]]]
  # Query = [[[1, 2]],
  #         [[30, 40]],
  #         [[1, 2]],
  #         [[1, 2]],
  #         [[100, 99]],
  #         [[1, 2]],
  #         [[1, 2]]]
  # print(nearest_neighbor(X, Query, 1))
  # print("Complete")
  test.main()
