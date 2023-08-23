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
_N_REAL_CAMERA_CHEST_FRAMES = 73
_N_REAL_CAMERA_HEAD_FRAMES = 60
_N_REAL_CAMERA_CHEST_FRAMES2 = 4
_N_SYNTHETIC_CAMERA_CHEST_FRAMES = 50
_BATCH_SIZE = 4
_PREFETCH_SIZE = 1
_SINGLE_IMAGE_SIZE = (1242, 2208, 3)
_BATCH_IMAGE_SIZE = (_BATCH_SIZE, 1242, 2208, 3)
_SINGLE_LABEL_SIZE = (1242, 2208, 1)
_BATCH_LABEL_SIZE = (_BATCH_SIZE, 1242, 2208, 1)


def _one_sample_from_each(*args: tf.data.Dataset):
  """Creates an iterator that yields exactly 1 sample from each dataset."""
  for dataset in args:
    yield iter(dataset).get_next()


def _parse_frame_id(frame_id: str):
  """Parses a frame id feature into a (sensor_id, frame_num)."""
  parts = frame_id.rsplit('/', 1)
  return (parts[0], int(parts[1]))


class TensorflowDatasetTest(tf.test.TestCase, parameterized.TestCase):

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

    for sample in _one_sample_from_each(trainset, testset):
      self.assertLen(sample, 8)
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

  def test_simple_synthetic_sanpo_dataset(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_real=False,
    ).to_tf_data()

    for sample in _one_sample_from_each(trainset, testset):
      self.assertLen(sample, 8)
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
      self.assertLen(sample, 8)
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

    for sample in _one_sample_from_each(trainset, testset):
      _verify_sample(sample)

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
      self.assertLen(sample, 12)
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

    for sample in _one_sample_from_each(trainset, testset):
      _verify_sample(sample)

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
      self.assertLen(sample, 12)
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

    for sample in _one_sample_from_each(trainset, testset):
      _verify_sample(sample)

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
      self.assertLen(sample, 15)
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

    for sample in _one_sample_from_each(trainset, testset):
      _verify_sample(sample)

  def test_synthetic_dataset(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_real=False,
        include_synthetic=True,
    ).to_tf_data()

    train_count = 0
    test_count = 0
    for _ in trainset:
      train_count += 1
    for _ in testset:
      test_count += 1
    self.assertEqual(train_count, _N_SYNTHETIC_CAMERA_CHEST_FRAMES)
    self.assertEqual(test_count, _N_SYNTHETIC_CAMERA_CHEST_FRAMES)

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
      self.assertLen(sample, 11)
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

    for sample in _one_sample_from_each(trainset, testset):
      _verify_sample(sample)

  def test_target_shape(self):
    trainset, testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.MULTITASK_SANPO_CONFIG,
        target_shape=[108, 192],
    ).to_tf_data()
    for sample in _one_sample_from_each(trainset, testset):
      expected_shape_img = (108, 192, 3)
      expected_shape_map = (108, 192, 1)
      tf.debugging.assert_shapes([
          (sample['image'], expected_shape_img),
          (sample['semantic_label'], expected_shape_map),
          (sample['instance_id'], expected_shape_map),
          (sample['metric_depth'], expected_shape_map),
          (sample['metric_depth_zed'], expected_shape_map),
      ])

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
    for sample in _one_sample_from_each(trainset, testset):
      self.assertLen(sample, 10)
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

  def test_video_mode(self):
    num_frames = 4

    trainset, _ = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.MULTITASK_SANPO_CONFIG,
        include_synthetic=False,
        dataset_view_mode=common.DatasetViewMode.STEREO_VIEW_VIDEO_MODE,
        num_video_frames=num_frames,
    ).to_tf_data()

    sample_count = 0
    for sample in trainset:
      self.assertEqual(
          sample[common.FEATURE_IMAGE].shape,
          tf.TensorShape((num_frames,) + _SINGLE_IMAGE_SIZE),
      )
      self.assertEqual(
          sample[common.FEATURE_IMAGE_RIGHT].shape,
          tf.TensorShape((num_frames,) + _SINGLE_IMAGE_SIZE),
      )
      self.assertEqual(
          sample[common.FEATURE_CAMERA_TRANSLATIONS].shape,
          tf.TensorShape((num_frames, 3)),
      )
      sample_count += 1

      # Check that all frames in the video clip are sequential and from the
      # same session/sensor.
      last_sensor_id = None
      last_frame_num = None
      for frame_id_tensor in sample[common.FEATURE_FRAME_ID]:
        frame_id = frame_id_tensor.numpy().decode('utf-8')
        sensor_id, frame_num = _parse_frame_id(frame_id)

        if last_sensor_id:
          self.assertEqual(
              sensor_id,
              last_sensor_id,
              'Video clip contains frame from different sensor',
          )
          self.assertEqual(
              frame_num,
              last_frame_num + 1,
              'Video clip contains non-sequential frames',
          )

        last_sensor_id = sensor_id
        last_frame_num = frame_num

    expected_sample_count = (
        int(_N_REAL_CAMERA_HEAD_FRAMES / num_frames)
        + int(_N_REAL_CAMERA_CHEST_FRAMES / num_frames)
        + int(_N_REAL_CAMERA_CHEST_FRAMES2 / num_frames)
    )
    self.assertEqual(sample_count, expected_sample_count)

  def test_single_split(self):
    testset = sanpo_dataset.SanpoDataset(
        self.sessions_dir,
        common.PANOPTIC_SANPO_CONFIG,
        include_synthetic=False,
        dataset_view_mode=common.DatasetViewMode.MONO_VIEW_FRAME_MODE,
    ).to_tf_data(split_name=sanpo_dataset.TEST_SPLITNAME)

    sample = next(_one_sample_from_each(testset))
    self.assertIn(common.FEATURE_IMAGE, sample)


if __name__ == '__main__':
  tf.test.main()
