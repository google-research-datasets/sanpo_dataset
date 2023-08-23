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

import dataclasses
import os
import tempfile
from absl import flags
from sanpo_dataset.lib import common
import tensorflow as tf

FLAGS = flags.FLAGS

_REAL_SESSIONS_PATH = 'third_party/py/sanpo_dataset/lib/testdata/sanpo-real'
_SYNTHETIC_SESSIONS_PATH = (
    'third_party/py/sanpo_dataset/lib/testdata/sanpo-synthetic'
)
_REAL_SESSION_NAME = 'real_session'
_SYNTHETIC_SESSION_NAME = 'synthetic_session'
_N_REAL_CAMERA_CHEST_VIDEOS = 73
_N_REAL_CAMERA_HEAD_VIDEOS = 73


class CommonTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.real_sessions_base_dir = os.path.join(
        FLAGS.test_srcdir, 'goo''gle3', _REAL_SESSIONS_PATH
    )
    self.real_session_dir = os.path.join(
        FLAGS.test_srcdir, 'goo''gle3', _REAL_SESSIONS_PATH, _REAL_SESSION_NAME
    )
    self.synthetic_sessions_base_dir = os.path.join(
        FLAGS.test_srcdir,
        'goo''gle3',
        _SYNTHETIC_SESSIONS_PATH,
    )
    self.synthetic_session_dir = os.path.join(
        FLAGS.test_srcdir,
        'goo''gle3',
        _SYNTHETIC_SESSIONS_PATH,
        _SYNTHETIC_SESSION_NAME,
    )

  def test_labelmap(self):
    labelmap_name_to_id = common.get_labelmap_name_to_id()
    self.assertLen(labelmap_name_to_id, 31)
    self.assertEqual(labelmap_name_to_id['unlabeled'], 0)
    self.assertEqual(labelmap_name_to_id['terrain'], 30)

    labelmap_id_to_name = common.get_labelmap_id_to_name()
    self.assertLen(labelmap_id_to_name, 31)
    self.assertEqual(labelmap_id_to_name[0], 'unlabeled')
    self.assertEqual(labelmap_id_to_name[30], 'terrain')

  def test_load_camera_poses_from_csv(self):
    camera_poses = common.load_camera_poses_from_csv(
        os.path.join(self.real_session_dir, 'camera_chest', 'camera_poses.csv')
    )
    self.assertLen(camera_poses, _N_REAL_CAMERA_CHEST_VIDEOS)
    self.assertFalse(camera_poses[0]['tracking_state'])
    self.assertArrayNear(
        camera_poses[0]['camera_translation_in_m'], [0.0, 0.0, 0.0], 1e-6
    )
    self.assertArrayNear(
        camera_poses[0]['camera_quaternions_right_handed_y_up'],
        [
            -0.0060849725268781185,
            0.04463198408484459,
            0.0008189158397726715,
            0.9989846348762512,
        ],
        1e-6,
    )
    self.assertTrue(camera_poses[-1]['tracking_state'])
    self.assertArrayNear(
        camera_poses[-1]['camera_translation_in_m'],
        [-0.8864306211471558, 0.6140850186347961, -5.516709327697754],
        1e-6,
    )
    self.assertArrayNear(
        camera_poses[-1]['camera_quaternions_right_handed_y_up'],
        [
            -0.0032249209471046925,
            0.027953239157795906,
            -0.05339173227548599,
            0.9981771111488342,
        ],
        1e-6,
    )

  def test_sanpo_real_session(self):
    self.assertLen(os.listdir(self.real_session_dir), 3)
    sanpo_session = common.SanpoSession(
        self.real_session_dir, dataclasses.replace(common.PANOPTIC_SANPO_CONFIG)
    )
    self.assertEqual(sanpo_session.name, 'real_session')
    self.assertEqual(sanpo_session.type, 'real')
    self.assertLen(sanpo_session.sensor_names, 2)
    self.assertEqual(
        sanpo_session.sensor_names, ['camera_chest', 'camera_head']
    )
    self.assertTrue(sanpo_session.has_metric_depth_zed)
    self.assertTrue(sanpo_session.has_segmentation_annotation)
    for sensor_name in sanpo_session.sensor_names:
      self.assertEqual(sanpo_session.lens_names(sensor_name), ['left', 'right'])
      _ = sanpo_session.camera_poses(sensor_name)
      self.assertEqual(
          sanpo_session.n_frames(sensor_name), _N_REAL_CAMERA_CHEST_VIDEOS
      )

  def test_frame_example(self):
    sanpo_session = common.SanpoSession(
        self.real_session_dir, dataclasses.replace(common.PANOPTIC_SANPO_CONFIG)
    )
    for sensor_name in sanpo_session.sensor_names:
      frame_example = common.FrameExample.from_session(
          sanpo_session, sensor_name, 0
      )
      self.assertTrue(frame_example.has_metric_depth)
      self.assertTrue(frame_example.has_metric_depth_zed)
      self.assertTrue(frame_example.has_right_lens)
      if sensor_name == 'camera_chest':
        self.assertTrue(frame_example.has_segmentation_mask)
      else:
        self.assertFalse(frame_example.has_segmentation_mask)

  def test_synthetic_frame_example(self):
    sanpo_session = common.SanpoSession(
        self.synthetic_session_dir,
        dataclasses.replace(common.PANOPTIC_SANPO_CONFIG),
    )
    self.assertSameElements(['camera_chest'], list(sanpo_session.sensor_names))
    for sensor_name in sanpo_session.sensor_names:
      frame_example = common.FrameExample.from_session(
          sanpo_session, sensor_name, 0
      )
      self.assertTrue(frame_example.has_metric_depth)
      self.assertFalse(frame_example.has_metric_depth_zed)
      self.assertFalse(frame_example.has_right_lens)
      self.assertTrue(frame_example.has_segmentation_mask)

  def test_sanpo_session_list(self):
    session_ids_file = tempfile.mktemp()
    with open(session_ids_file, 'w') as fileptr:
      fileptr.write('real_session')
    sanpo_session_list = common.SanpoSessionList(
        self.real_sessions_base_dir,
        config=common.PANOPTIC_SANPO_CONFIG,
        session_ids_or_ids_file=session_ids_file,
    )
    self.assertLen(list(sanpo_session_list.all_sessions), 1)
    self.assertLen(
        list(sanpo_session_list.get_valid_sessions()),
        1,
    )

    sanpo_session_list = common.SanpoSessionList(
        self.real_sessions_base_dir,
        config=common.PANOPTIC_SANPO_CONFIG,
        session_ids_or_ids_file=['real_session'],
    )
    self.assertLen(list(sanpo_session_list.all_sessions), 1)
    self.assertLen(
        list(sanpo_session_list.get_valid_sessions()),
        1,
    )

  def test_iter_samples_panoptic_frame_mode(self):
    sanpo_session = common.SanpoSession(
        self.real_session_dir, dataclasses.replace(common.PANOPTIC_SANPO_CONFIG)
    )
    count = 0
    for example in sanpo_session.itersamples():
      self.assertLen(example, 6)
      count += 1
    self.assertEqual(count, _N_REAL_CAMERA_CHEST_VIDEOS)

  def test_iter_samples_all_optional_frame_mode(self):
    config = dataclasses.replace(common.PANOPTIC_SANPO_CONFIG)
    config.feature_panoptic_mask = common.FeatureFilterOption.INCLUDE
    config.feature_metric_depth = common.FeatureFilterOption.INCLUDE
    config.feature_metric_depth_zed = common.FeatureFilterOption.INCLUDE
    config.feature_camera_pose = common.FeatureFilterOption.INCLUDE
    sanpo_session = common.SanpoSession(self.real_session_dir, config)
    count = 0
    for example in sanpo_session.itersamples():
      self.assertLen(example, 13)
      count += 1
    self.assertEqual(
        count, _N_REAL_CAMERA_CHEST_VIDEOS + _N_REAL_CAMERA_HEAD_VIDEOS
    )

  def test_iter_samples_stereo_mode(self):
    config = dataclasses.replace(common.DEPTH_SANPO_CONFIG)
    config.dataset_view_mode = common.DatasetViewMode.STEREO_VIEW_FRAME_MODE
    sanpo_session = common.SanpoSession(self.real_session_dir, config)
    count = 0
    for example in sanpo_session.itersamples():
      self.assertIn('image', example)
      self.assertIn('image_right', example)
      count += 1
    self.assertEqual(
        count, _N_REAL_CAMERA_CHEST_VIDEOS + _N_REAL_CAMERA_HEAD_VIDEOS
    )


if __name__ == '__main__':
  tf.test.main()
