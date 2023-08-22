# Copyright 2023 The sanpo_dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for common.py."""

import itertools
import os
from absl import flags
from absl.testing import parameterized
from sanpo_dataset.lib import common
from sanpo_dataset.lib import tensorflow_dataset as sanpo_dataset
import tensorflow as tf

FLAGS = flags.FLAGS

_SESSIONS_PATH = 'third_party/py/sanpo_dataset/lib/testdata'
_REAL_SESSION_NAME = 'real_session'
_SYNTHETIC_SESSION_NAME = 'synthetic_session'
_N_REAL_CAMERA_CHEST_VIDEOS = 73
_N_REAL_CAMERA_HEAD_VIDEOS = 73
_BATCH_SIZE = 4
_PREFETCH_SIZE = 1
_SINGLE_IMAGE_SIZE = (1242, 2208, 3)
_BATCH_IMAGE_SIZE = (_BATCH_SIZE, 1242, 2208, 3)
_SINGLE_LABEL_SIZE = (1242, 2208, 1)
_SINGLE_CRESTEREO_DEPTH_SIZE = (720, 1280, 1)
_BATCH_CRESTEREO_DEPTH_SIZE = (_BATCH_SIZE, 720, 1280, 1)
_BATCH_LABEL_SIZE = (_BATCH_SIZE, 1242, 2208, 1)


class CommonTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.sessions_dir = os.path.join(
        FLAGS.test_srcdir, 'goo''gle3', _SESSIONS_PATH
    )

  def test_simple_real_sanpo_dataset(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_synthetic=False,
    ).to_tf_data()
    for sample in trainset:
      self.assertLen(sample, 7)
      self.assertEqual(sample['session_type'], 'real')
      self.assertSequenceEqual(
          sample['camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
          .numpy()
          .tolist(),
          [
              0.8938735127449036,
              1.5891084671020508,
              0.5139687061309814,
              0.527898371219635,
          ],
      )
      self.assertAlmostEqual(
          sample['camera_baseline_in_m'].numpy(), 0.119918846, 3
      )
      break
    for sample in testset:
      self.assertLen(sample, 7)
      self.assertEqual(sample['session_type'], 'real')
      self.assertSequenceEqual(
          sample['camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
          .numpy()
          .tolist(),
          [
              0.8938735127449036,
              1.5891084671020508,
              0.5139687061309814,
              0.527898371219635,
          ],
      )
      self.assertAlmostEqual(
          sample['camera_baseline_in_m'].numpy(), 0.119918846, 3
      )
      break

  def test_simple_synthetic_sanpo_dataset(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_real=False,
    ).to_tf_data()
    for sample in trainset:
      self.assertLen(sample, 7)
      self.assertEqual(sample['session_type'], 'synthetic')
      self.assertSequenceEqual(
          sample['camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
          .numpy()
          .tolist(),
          [
              0.8669397830963135,
              1.5412262678146362,
              0.486392080783844,
              0.5275930166244507,
          ],
      )
      self.assertEqual(sample['camera_baseline_in_m'], 0.0)
      break
    for sample in testset:
      self.assertLen(sample, 7)
      self.assertEqual(sample['session_type'], 'synthetic')
      self.assertSequenceEqual(
          sample['camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
          .numpy()
          .tolist(),
          [
              0.8669397830963135,
              1.5412262678146362,
              0.486392080783844,
              0.5275930166244507,
          ],
      )
      self.assertEqual(sample['camera_baseline_in_m'], 0.0)
      break

  @parameterized.parameters([False, True])
  def test_get_sanpo_panoptic_dataset_frame_mode(self, batch_mode: bool):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        builder_config=common.PANOPTIC_SANPO_CONFIG,
    ).to_tf_data()

    expected_image_size = _SINGLE_IMAGE_SIZE
    expected_label_size = _SINGLE_LABEL_SIZE
    if batch_mode:
      trainset = trainset.batch(_BATCH_SIZE)
      trainset = trainset.prefetch(_PREFETCH_SIZE)
      testset = testset.batch(_BATCH_SIZE)
      testset = testset.prefetch(_PREFETCH_SIZE)
      expected_image_size = _BATCH_IMAGE_SIZE
      expected_label_size = _BATCH_LABEL_SIZE

    def _verify_sample(sample):
      self.assertLen(sample, 7)
      tf.debugging.assert_shapes([
          (sample['image'], expected_image_size),
          (sample['semantic_label'], expected_label_size),
          (sample['instance_id'], expected_label_size),
      ])
      self.assertAllInRange(sample['image'], 0, 1)
      if batch_mode:
        self.assertTrue(all(sample['has_panoptic_label']))
        tf.debugging.assert_shapes([
            (sample['has_panoptic_label'], (_BATCH_SIZE,)),
        ])
      else:
        self.assertTrue(sample['has_panoptic_label'])

    for sample in trainset:
      _verify_sample(sample)
      break

    for sample in testset:
      _verify_sample(sample)
      break

  @parameterized.parameters([False, True])
  def test_get_sanpo_panoptic_dataset_frame_mode_with_zed_depth(
      self, real_only: bool
  ):
    config = common.PANOPTIC_SANPO_CONFIG.replace()
    config.feature_metric_depth = common.FeatureFilterOption.INCLUDE
    config.feature_metric_depth_zed = common.FeatureFilterOption.INCLUDE

    # Either real or synthetic.
    if real_only:
      config.include_real = True
      config.include_synthetic = False
    else:
      config.include_real = False
      config.include_synthetic = True

    # Panoptic dataset with camera pose.
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir, config, target_shape=_SINGLE_LABEL_SIZE[:2]
    ).to_tf_data()

    expected_image_size = _SINGLE_IMAGE_SIZE
    expected_label_size = _SINGLE_LABEL_SIZE
    expected_metric_depth_size = _SINGLE_LABEL_SIZE

    def _verify_sample(sample):
      self.assertLen(sample, 11)
      tf.debugging.assert_shapes([
          (sample['image'], expected_image_size),
          (sample['semantic_label'], expected_label_size),
          (sample['instance_id'], expected_label_size),
          (sample['metric_depth'], expected_metric_depth_size),
          (sample['metric_depth_zed'], expected_label_size),
      ])
      self.assertTrue(sample['has_panoptic_label'])
      if real_only:
        self.assertEqual(sample['session_type'], 'real')
        self.assertTrue(sample['has_metric_depth_zed'])
      else:
        self.assertEqual(sample['session_type'], 'synthetic')
        self.assertFalse(sample['has_metric_depth_zed'])

    for sample in trainset:
      _verify_sample(sample)
      break

    for sample in testset:
      _verify_sample(sample)
      break

  @parameterized.parameters([False, True])
  def test_get_sanpo_depth_dataset_frame_mode(self, real_mode: bool):
    config = common.DEPTH_SANPO_CONFIG.replace()

    # Either real or synthetic.
    if real_mode:
      config.include_real = True
      config.include_synthetic = False
    else:
      config.include_real = False
      config.include_synthetic = True

    # Panoptic dataset with camera pose.
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir, config, target_shape=_SINGLE_LABEL_SIZE[:2]
    ).to_tf_data()

    expected_image_size = _SINGLE_IMAGE_SIZE
    expected_label_size = _SINGLE_LABEL_SIZE
    expected_metric_depth_size = _SINGLE_LABEL_SIZE

    def _verify_sample(sample):
      self.assertLen(sample, 11)
      tf.debugging.assert_shapes([
          (sample['image'], expected_image_size),
          (sample['metric_depth'], expected_metric_depth_size),
          (sample['metric_depth_zed'], expected_label_size),
      ])
      self.assertIn('has_metric_depth_zed', sample.keys())
      if real_mode:
        self.assertEqual(sample['session_type'], 'real')
      else:
        self.assertEqual(sample['session_type'], 'synthetic')
        self.assertFalse(sample['has_metric_depth_zed'])

    for sample in trainset:
      _verify_sample(sample)
      break

    for sample in testset:
      _verify_sample(sample)
      break

  @parameterized.parameters([False, True])
  def test_get_sanpo_multitask_dataset_frame_mode(self, real_mode: bool):
    config = common.MULTITASK_SANPO_CONFIG.replace()

    # Either real or synthetic.
    if real_mode:
      config.include_real = True
      config.include_synthetic = False
    else:
      config.include_real = False
      config.include_synthetic = True

    # Panoptic dataset with camera pose.
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        config,
        target_shape=_SINGLE_IMAGE_SIZE[:2],
    ).to_tf_data()

    expected_image_size = _SINGLE_IMAGE_SIZE
    expected_label_size = _SINGLE_LABEL_SIZE
    expected_metric_depth_size = _SINGLE_LABEL_SIZE

    def _verify_sample(sample):
      self.assertLen(sample, 14)
      tf.debugging.assert_shapes([
          (sample['image'], expected_image_size),
          (sample['semantic_label'], expected_label_size),
          (sample['instance_id'], expected_label_size),
          (sample['metric_depth'], expected_metric_depth_size),
          (sample['metric_depth_zed'], expected_label_size),
      ])
      self.assertIn('metric_depth', sample.keys())
      self.assertIn('has_metric_depth_zed', sample.keys())
      self.assertIn('has_panoptic_label', sample.keys())
      self.assertIn('has_metric_depth', sample.keys())
      if real_mode:
        self.assertEqual(sample['session_type'], 'real')
      else:
        self.assertEqual(sample['session_type'], 'synthetic')

    train_count = test_count = 0
    for sample in trainset:
      train_count += 1
      _verify_sample(sample)
      break

    for sample in testset:
      test_count += 1
      _verify_sample(sample)
      break
    self.assertGreater(train_count, 0, 'no training data')
    self.assertGreater(test_count, 0, 'no test data')

  def test_synthetic_dataset(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_real=False,
        include_synthetic=True,
    ).to_tf_data()

    train_count = test_count = 0
    # pylint: disable=unused-variable
    for sample in trainset:
      train_count += 1
    for sample in testset:
      test_count += 1
    self.assertGreater(train_count, 0, 'no synthetic training data')
    self.assertGreater(test_count, 0, 'no synthetic test data')
    # pylint: enable=unused-variable

  @parameterized.parameters([False, True])
  def test_get_sanpo_dataset_frame_mode_camera_pose(self, batch_mode: bool):
    # Panoptic dataset with camera pose.
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        feature_camera_pose=common.FeatureFilterOption.INCLUDE,
    ).to_tf_data()

    expected_image_size = _SINGLE_IMAGE_SIZE
    expected_label_size = _SINGLE_LABEL_SIZE
    if batch_mode:
      trainset = trainset.batch(_BATCH_SIZE)
      trainset = trainset.prefetch(_PREFETCH_SIZE)
      testset = testset.batch(_BATCH_SIZE)
      testset = testset.prefetch(_PREFETCH_SIZE)
      expected_image_size = _BATCH_IMAGE_SIZE
      expected_label_size = _BATCH_LABEL_SIZE

    def _verify_sample(sample):
      self.assertLen(sample, 10)
      tf.debugging.assert_shapes([
          (sample['image'], expected_image_size),
          (sample['semantic_label'], expected_label_size),
          (sample['instance_id'], expected_label_size),
      ])
      if batch_mode:
        self.assertTrue(all(sample['has_panoptic_label']))
        tf.debugging.assert_shapes([
            (sample['has_panoptic_label'], (_BATCH_SIZE,)),
        ])
        tf.debugging.assert_shapes([
            (sample['tracking_state'], (_BATCH_SIZE,)),
            (sample['camera_translation_in_m'], (_BATCH_SIZE, 3)),
            (sample['camera_quaternions_right_handed_y_up'], (_BATCH_SIZE, 4)),
        ])
      else:
        self.assertTrue(sample['has_panoptic_label'])
        tf.debugging.assert_shapes([
            (sample['tracking_state'], ()),
            (sample['camera_translation_in_m'], (3,)),
            (sample['camera_quaternions_right_handed_y_up'], (4,)),
        ])

    for sample in trainset:
      _verify_sample(sample)
      break

    for sample in testset:
      _verify_sample(sample)
      break

  def test_target_shape(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.MULTITASK_SANPO_CONFIG,
        target_shape=[108, 192],
    ).to_tf_data()
    count = 0
    for sample in itertools.chain(trainset, testset):
      count += 1
      expected_shape_img = (108, 192, 3)
      expected_shape_map = (108, 192, 1)
      tf.debugging.assert_shapes([
          (sample['image'], expected_shape_img),
          (sample['semantic_label'], expected_shape_map),
          (sample['instance_id'], expected_shape_map),
          (sample['metric_depth'], expected_shape_map),
          (sample['metric_depth_zed'], expected_shape_map),
      ])
    self.assertGreater(count, 0)

  def test_invalid_target_shape(self):
    with self.assertRaises(ValueError):
      trainset, testset = sanpo_dataset.SanpoDataset(
          self.sessions_dir,
          common.MULTITASK_SANPO_CONFIG,
          target_shape=[123, 456],
      ).to_tf_data()
      for _ in itertools.chain(trainset, testset):
        pass


def test_stereo_frame_mode(self):
  trainset, testset = sanpo_dataset.SanpoDataset(
      self.sessions_dir,
      common.PANOPTIC_SANPO_CONFIG,
      include_synthetic=False,
      dataset_view_mode=common.DatasetViewMode.STEREO_VIEW_FRAME_MODE,
  ).to_tf_data()
  for sample in trainset:
    self.assertLen(sample, 9)
    self.assertIn('image', sample)
    self.assertIn('image_right', sample)
    self.assertSequenceEqual(
        sample['camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
        .numpy()
        .tolist(),
        [
            0.8938735127449036,
            1.5891084671020508,
            0.5139687061309814,
            0.527898371219635,
        ],
    )
    self.assertSequenceEqual(
        sample['camera_right_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy']
        .numpy()
        .tolist(),
        [
            0.8938735127449036,
            1.5891084671020508,
            0.5139687061309814,
            0.527898371219635,
        ],
    )
    self.assertAlmostEqual(
        sample['camera_baseline_in_m'].numpy(), 0.119918846, 3
    )
    break

  for sample in testset:
    self.assertLen(sample, 9)
    self.assertIn('image', sample)
    self.assertIn('image_right', sample)
    break


if __name__ == '__main__':
  tf.test.main()
