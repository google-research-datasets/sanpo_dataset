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

"""Common operations for SANPO dataset, used by TF and PyTorch."""

import csv
import dataclasses
import enum
import functools
import json
import os
import pathlib
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np

DEFAULT_SENSOR_NAME = 'camera_head'
SENSOR_NAMES = [
    'camera_head',  # also used for synthetic
    'camera_chest',
]
LENS_NAMES = [
    'left',
    'right',
]
LEFT_LENS_NAME = 'left'
RIGHT_LENS_NAME = 'right'

DEPTH_DIRNAME = 'depth_maps'
ZED_DEPTH_DIRNAME = 'zed_depth_maps'
SEGMENTATION_MASK_DIRNAME = 'segmentation_masks'
RGB_FRAME_DIRNAME = 'video_frames'
FILENAME_RGB = '{frame_num:06d}.png'
FILENAME_DEPTH = '{frame_num:06d}.float16.gz'
FILENAME_SEMANTIC = '{frame_num:06d}.png'
IMAGE_FILENAME_EXTENSION = '.png'
IMAGE_MASK_FILENAME_EXTENSION = '.png'
DEPTH_FILENAME_EXTENSION = '.npz'
CAMERA_POSES_CSV_FILENAME = 'camera_poses.csv'


FEATURE_SESSION_TYPE = 'session_type'
FEATURE_IMAGE = 'image'
FEATURE_IMAGE_RIGHT = 'image_right'
FEATURE_METRIC_DEPTH_LABEL = 'metric_depth'
FEATURE_HAS_METRIC_DEPTH_LABEL = 'has_metric_depth'
FEATURE_PANOPTIC_MASK_LABEL = 'panoptic_label'
FEATURE_HAS_PANOPTIC_MASK_LABEL = 'has_panoptic_label'
FEATURE_SEMANTIC_LABEL = 'semantic_label'
FEATURE_INSTANCE_ID = 'instance_id'
FEATURE_METRIC_DEPTH_ZED_LABEL = 'metric_depth_zed'
FEATURE_HAS_METRIC_DEPTH_ZED_LABEL = 'has_metric_depth_zed'
FEATURE_TRACKING_STATE = 'tracking_state'
FEATURE_CAMERA_TRANSLATIONS = 'camera_translation_in_m'
FEATURE_CAMERA_QUATERNIONS = 'camera_quaternions_right_handed_y_up'
FEATURE_CAMERA_INTRINSICS = (
    'camera_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy'
)
FEATURE_CAMERA_RIGHT_INTRINSICS = (
    'camera_right_intrinsics_as_ratios_of_image_size_fx_fy_cx_cy'
)
FEATURE_CAMERA_BASELINE_IN_METERS = 'camera_baseline_in_m'

CAMERA_TRACKING_STATE_READY = 'TrackingState.READY'
CAMERA_TRACKING_STATE_NOT_READY = 'TrackingState.NOT_READY'


# pylint: disable=unreachable
def wrapped_open(filename, mode):
  """Return a file object.

  (Dispatches between internal gfile.Open and external open())

  Args:
    filename: Path to file to open
    mode: Either 'r' or 'rb'

  Returns:
    An object that can be read()
  """
  # within candled fog,
  #   in nacreous light,
  # some secrets are hidden
  #   away from mere sight

  # mere echoes remain now,
  #   of something once real,
  # source coude now enshrouded
  #   by copybara's seal
  return open(filename, mode)


def wrapped_path(path: Any) -> Any:
  """Return a path object."""
  return pathlib.Path(path)


# pylint: enable=unreachable


class DatasetViewMode(enum.Enum):

  """Configures which data to include in each sample.

  STEREO_VIEW_FRAME_MODE: Stereo view in frame mode. Each sample contains both
    the left and right camera frame plus the corresponding labels.
  STEREO_VIEW_VIDEO_MODE: Stereo view in video mode. Each sample contains both
    the left and right camera video plus the corresponding labels.
    The number of frames in the video will provided in the SANPO_CONFIG.
  MONO_VIEW_FRAME_MODE: Mono view in frame mode. Each sample contains one of the
    the left or right camera frame plus the corresponding labels. Note right
    camera frame has no semantic or depth labels. Whether to include right
    camera frame at all can be defined in the SANPO config.
   MONO_VIEW_VIDEO_MODE: Mono view in video mode. Each sample contains one of
    the left or right camera video plus the corresponding labels. Note right
    camera video has no semantic or depth labels. Whether to include right
    camera video at all can be defined in the SANPO config.
  """

  STEREO_VIEW_FRAME_MODE = 'stereo_view_frame_mode'
  STEREO_VIEW_VIDEO_MODE = 'stereo_view_video_mode'
  MONO_VIEW_FRAME_MODE = 'mono_view_frame_mode'
  MONO_VIEW_VIDEO_MODE = 'mono_view_video_mode'


class FeatureFilterOption(enum.Enum):
  """Provides options for user filtering of SANPO attributes.

  Some SANPO samples have segmentation masks, some do not. Some have ZED depth,
  some do not. This filter lets users specify whether to load this data,
  and whether it is required for their application. The below values are only
  for feature inclusion or exclusion. If the mandatory feature is not present
  then that sample is excluded. Please note that this exclusion is different
  from what sessions to include to begin with. You can control that through the
  `session_ids_or_ids_file` argument in the SanpoSessionList.

  MANDATORY: Returned sample must include this feature.
  INCLUDE: Include this feature if available.
  EXCLUDE: Exclude the feature.
  """

  MANDATORY = 'mandatory'
  INCLUDE = 'include'
  EXCLUDE = 'exclude'

  def is_mandatory(self) -> bool:
    """Determine whether to mandatorily include this feature."""
    return self == FeatureFilterOption.MANDATORY

  def to_include(self) -> bool:
    """Determine whether to mandatorily include this feature."""
    return (
        self == FeatureFilterOption.MANDATORY
        or self == FeatureFilterOption.INCLUDE
    )

  def to_exclude(self) -> bool:
    """Determine whether to exclude this feature."""
    return self == FeatureFilterOption.EXCLUDE

  def test_sample_flag(self, flag: bool) -> bool:
    """Determine whether this flag matches the input flag."""
    if self == flag:
      return True
    return False


@dataclasses.dataclass
class SanpoConfig:
  """Configuration for SANPO.

  Users can filter the dataset to include/exclude metric depth, segmentation
  masks, etc. Each of these features may additionally be marked as 'REQUIRED',
  which signals that only examples that have that feature will be included.

  Attributes:
    include_real (bool): Whether to include real images
    include_synthetic (bool): Whether to include synthetic images
    feature_metric_depth (FeatureFilterOption): Whether to include metric depth
    feature_metric_depth_zed (FeatureFilterOption): Whether to include metric
      depth data from the ZED API (SANPO-Real only)
    feature_panoptic_mask (FeatureFilterOption): Whether to include panoptic
      instance-level segmentation masks
    feature_camera_pose (FeatureFilterOption): Whether to include camera pose
    dataset_view_mode (DatasetViewMode): What view of the sessions to provide.
    target_shape (Tuple[h,w]): Optional shape to resize all maps to. Note:
      This code does not do cropping. If target_shape=None, then
      RGB images and depth maps may have different sizes.
    num_video_frames: Optional. Required when using the video modes.
    video_frame_stride: Optional. Default is 1. Defines stride to generate video
      samples.
    shuffle_buffer_size: Size of shuffle buffer.
  """

  include_real: bool
  include_synthetic: bool
  feature_metric_depth: FeatureFilterOption
  feature_metric_depth_zed: FeatureFilterOption
  feature_panoptic_mask: FeatureFilterOption
  feature_camera_pose: FeatureFilterOption
  dataset_view_mode: DatasetViewMode
  target_shape: Optional[Tuple[int, int]] = None
  num_video_frames: Optional[int] = None
  video_frame_stride: Optional[int] = None
  shuffle_buffer_size: int = 10240

  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


PANOPTIC_SANPO_CONFIG = SanpoConfig(
    include_real=True,
    include_synthetic=True,
    feature_metric_depth=FeatureFilterOption.EXCLUDE,
    feature_metric_depth_zed=FeatureFilterOption.EXCLUDE,
    feature_panoptic_mask=FeatureFilterOption.MANDATORY,
    feature_camera_pose=FeatureFilterOption.EXCLUDE,
    dataset_view_mode=DatasetViewMode.MONO_VIEW_FRAME_MODE,
)
DEPTH_SANPO_CONFIG = SanpoConfig(
    include_real=True,
    include_synthetic=True,
    feature_metric_depth=FeatureFilterOption.MANDATORY,
    feature_metric_depth_zed=FeatureFilterOption.INCLUDE,
    feature_panoptic_mask=FeatureFilterOption.EXCLUDE,
    feature_camera_pose=FeatureFilterOption.INCLUDE,
    dataset_view_mode=DatasetViewMode.MONO_VIEW_FRAME_MODE,
)
MULTITASK_SANPO_CONFIG = SanpoConfig(
    include_real=True,
    include_synthetic=True,
    feature_metric_depth=FeatureFilterOption.INCLUDE,
    feature_metric_depth_zed=FeatureFilterOption.INCLUDE,
    feature_panoptic_mask=FeatureFilterOption.INCLUDE,
    feature_camera_pose=FeatureFilterOption.INCLUDE,
    dataset_view_mode=DatasetViewMode.MONO_VIEW_FRAME_MODE,
)


def load_camera_poses_from_csv(
    csv_file: Union[str, pathlib.Path]
) -> List[Mapping[str, Union[bool, np.ndarray]]]:
  """Loads and returns camera poses from a csv."""
  camera_poses = []
  with wrapped_open(csv_file, 'r') as fileptr:
    reader = csv.DictReader(fileptr)
    for line in reader:
      # tracking_state,pos_x,pos_y,pos_z,q_x,q_y,q_z,q_w
      tracking_state = False
      if line['tracking_state'] == CAMERA_TRACKING_STATE_READY:
        tracking_state = True

      camera_poses.append({
          FEATURE_TRACKING_STATE: tracking_state,
          FEATURE_CAMERA_TRANSLATIONS: np.array(
              [line['pos_x'], line['pos_y'], line['pos_z']],
              dtype=np.float32,
          ),
          FEATURE_CAMERA_QUATERNIONS: np.array(
              [line['q_x'], line['q_y'], line['q_z'], line['q_w']],
              dtype=np.float32,
          ),
      })
  return camera_poses


@dataclasses.dataclass
class SanpoSessionList:
  """A dataclass consisting of a list of SANPO sessions."""

  base_path: pathlib.Path
  session_ids: List[str]

  def __init__(
      self,
      sessions_dir_path: Union[str, pathlib.Path],
      *,
      config: SanpoConfig,
      session_ids_or_ids_file: Optional[Union[List[str], str]] = None,
  ):
    super().__init__()
    self.base_path = wrapped_path(sessions_dir_path)
    self.config = config

    self.session_ids = []
    if session_ids_or_ids_file is not None:
      if isinstance(session_ids_or_ids_file, list):
        self.session_ids = session_ids_or_ids_file
      else:
        with wrapped_open(
            os.path.join(session_ids_or_ids_file), 'r'
        ) as fileptr:
          for session_id in fileptr:
            self.session_ids.append(session_id.strip())
    else:
      for session_path in self.base_path.iterdir():
        if session_path.is_dir():
          self.session_ids.append(session_path.name)

    # Raise error if any of sessions' folder don't exist.
    for session_id in self.session_ids:
      session_path = self.base_path / session_id
      if not session_path.is_dir():
        raise ValueError(
            f'SanpoSessionList: Session path does not exisit: {session_path}'
        )

  @property
  def all_sessions(self) -> Iterator['SanpoSession']:
    for session_id in self.session_ids:
      session_path = self.base_path / session_id
      yield SanpoSession(session_path, self.config)

  # pylint: disable=g-doc-return-or-yield
  def get_valid_sessions(
      self,
  ) -> Iterator['SanpoSession']:
    """Returns list of session paths which has segmentation annotation.

    Yields: Matching session
    """
    # TODO(sagarwaghmare, kwilber): Filter sessions based on the self.config.
    for session_id in self.session_ids:
      session_path = self.base_path / session_id
      sanpo_session = SanpoSession(session_path, self.config)

      if self.config.feature_panoptic_mask.is_mandatory():
        if not sanpo_session.has_segmentation_annotation:
          continue

      if self.config.feature_metric_depth_zed.is_mandatory():
        if not sanpo_session.has_metric_depth_zed:
          continue

      # TODO(sagarwaghmare, kwilber): Check all other requirements from the
      # config. And skip the sessions which don't satisfy them.

      yield sanpo_session

  # pylint: enable=g-doc-return-or-yield


@dataclasses.dataclass
class SanpoSession:
  """A single SANPO recording session.

  Contains up to four streams of RGB video.
  """

  base_path: pathlib.Path
  config: SanpoConfig

  def __init__(
      self, session_dir_path: Union[str, pathlib.Path], config: SanpoConfig
  ):
    super().__init__()
    self.base_path = wrapped_path(session_dir_path)
    self.config = config

  @property
  def name(self):
    return self.base_path.name

  @functools.cached_property
  def sensor_names(self) -> List[str]:
    self._sensor_names = []
    for entry in self.base_path.iterdir():
      if entry.is_dir():
        self._sensor_names.append(entry.name)
    self._sensor_names.sort()
    return self._sensor_names

  @property
  def description(self):
    if not hasattr(self, '_description'):
      with wrapped_open(self.base_path / 'description.json', 'r') as fileptr:
        self._description = json.load(fileptr)
    return self._description

  @property
  def type(self):
    return self.description['session_type']

  @property
  def video_metadata(self):
    return self.description['session_video_metadata']

  def camera_details(self, sensor_name: str) -> Mapping[str, Any]:
    for camera_location, camera_details in zip(
        self.description['session_camera_location'],
        self.description['session_camera_details'],
    ):
      if sensor_name == camera_location:
        return camera_details
    return {}

  @functools.cached_property
  def has_metric_depth_zed(self) -> bool:
    """Returns true if session has Zed depth."""
    frame_example_head_camera = FrameExample.from_session(
        self, 'camera_head', 0
    )
    if frame_example_head_camera is not None:
      return frame_example_head_camera.has_metric_depth_zed
    return False

  @functools.cached_property
  def has_segmentation_annotation(self) -> bool:
    """Returns true if session has segmentation annotation else false."""
    for sensor_name in self.sensor_names:
      self._frame_example = FrameExample.from_session(self, sensor_name, 0)
      if (
          self._frame_example is not None
          and self._frame_example.has_segmentation_mask
      ):
        return True
    return False

  def lens_names(self, sensor_name: str) -> List[str]:
    """Returns lens names in the session's sensor."""

    # Just compute once.
    if not hasattr(self, '_sensor_lens_names'):
      self._sensor_lens_names = {}
      for sensor_name in self.sensor_names:
        sensor_dir = self.base_path / sensor_name
        lens_names = []
        for lens_name in sensor_dir.iterdir():
          if lens_name.is_dir():
            lens_names.append(lens_name.name)
        lens_names.sort()
        self._sensor_lens_names[sensor_name] = lens_names

    return self._sensor_lens_names[sensor_name]

  def camera_poses(
      self, sensor_name: str
  ) -> List[Mapping[str, Union[bool, np.ndarray]]]:
    """Returns camera poses corresponding the session's sensor."""

    # Just load once.
    if not hasattr(self, '_sensor_camera_poses'):
      self._sensor_camera_poses = {}
      for sensor_name in self.sensor_names:
        sensor_dir = self.base_path / sensor_name
        csv_file = sensor_dir / CAMERA_POSES_CSV_FILENAME
        self._sensor_camera_poses[sensor_name] = load_camera_poses_from_csv(
            csv_file
        )

    return self._sensor_camera_poses[sensor_name]

  def camera_intrinsics(
      self, sensor_name: str, lens_name: str
  ) -> Tuple[float, float, float, float, float]:
    """Returns camera intrinsics for the given sensor and lens.

    Args:
      sensor_name: Name of the sensor / camera rig.
      lens_name: Lens of the camera. (left/right)

    Returns:
      An array,
        [fx_ratio, fy_ratio, cx_ratio, cy_ratio, baseline_in_meters],
      where
        fx_ratio: the focal width, in terms of ratio of the image width
        fy_ratio: the focal height, in terms of ratio of the image height
        cx_ratio: principal point's X coordinate, in fractions of image width
        cy_ratio: principal point's Y coordinate, in fractions of image width
        baseline_in_meters: distance between left and right lens. 0 for
          monocular camera rigs.
    """
    if not hasattr(self, '_camera_intrinsics'):
      self._camera_intrinsics = {}
    if (sensor_name, lens_name) not in self._camera_intrinsics:
      camera_details = self.camera_details(sensor_name)
      camera_params = {
          LEFT_LENS_NAME: camera_details.get('left_camera_params'),
          RIGHT_LENS_NAME: camera_details.get('right_camera_params'),
      }[lens_name]
      fx_ratio = camera_params['fx'] / camera_params['image_width']
      fy_ratio = camera_params['fy'] / camera_params['image_height']
      cx_ratio = camera_params['cx'] / camera_params['image_width']
      cy_ratio = camera_params['cy'] / camera_params['image_height']
      if 'stereo_transform' in camera_details:
        baseline_in_mm = camera_details['stereo_transform']['coeff'][3]
      else:
        baseline_in_mm = 0.0
      self._camera_intrinsics[(sensor_name, lens_name)] = (
          fx_ratio,
          fy_ratio,
          cx_ratio,
          cy_ratio,
          baseline_in_mm,
      )
    return self._camera_intrinsics[(sensor_name, lens_name)]

  def n_frames(self, sensor_name: str) -> int:
    """Returns number of frames in the session's sensor."""
    # Just compute once.
    if not hasattr(self, '_sensor_n_frames'):
      self._sensor_n_frames = {}
      for sensor_name in self.sensor_names:
        ex = FrameExample.from_session(self, sensor_name, 0)
        if ex is None:
          self._sensor_n_frames[sensor_name] = 0
        else:
          lens_name = self.lens_names(sensor_name)[0]
          path = ex.rgb_filename(lens_name)
          files = list(path.parent.iterdir())
          files = [
              file
              for file in files
              if str(file).endswith(IMAGE_FILENAME_EXTENSION)
          ]
          self._sensor_n_frames[sensor_name] = len(files)
    return self._sensor_n_frames[sensor_name]

  def _skip_frame_example(
      self, ex: 'FrameExample', lens_name: str, stereo_mode: bool
  ) -> bool:
    """Checks whether to skip the sample."""
    if stereo_mode and not ex.has_right_lens:
      # Skip example if stereo mode is requested but there is no right lens.
      return True

    if (
        self.config.feature_panoptic_mask.is_mandatory()
        and not ex.segmentation_mask_filename(lens_name).exists()
    ):
      # Skip the frame example as it does not have the mandatory
      # panoptic segmentation annotation.
      return True

    if (
        self.config.feature_metric_depth.is_mandatory()
        and not ex.has_metric_depth
    ):
      # Skip the frame example as it does not have the
      # mandatory metric depth map.
      return True

    if (
        self.config.feature_metric_depth_zed.is_mandatory()
        and not ex.metric_depth_zed_filename(lens_name).exists()
    ):
      # Skip the frame example as it does not have the mandatory
      # zed depth.
      return True

    if (
        self.config.feature_camera_pose.is_mandatory()
        and not ex.has_camera_pose
    ):
      return True
    return False

  def itersamples_frame_mode(
      self,
      stereo_mode: bool = False,
  ) -> Iterator[Mapping[str, Any]]:
    """Yield sample dictionary based on the view and config in frame mode."""
    for sensor_name in self.sensor_names:
      for frame_num in range(self.n_frames(sensor_name)):
        ex = FrameExample.from_session(self, sensor_name, frame_num)
        if ex is None:
          continue

        # TODO(sagarwaghmare, kwilber): Filter frames based on the SANPO CONFIG.
        # "session_type": [b,] a string or enum - real or synthetic
        # "image": [b, h, w, 3],
        # "semantic_labels": [b,h,w,1],
        # "instance_ids": [b, h, w, 1],
        # "camera_translation": [b,3]
        # "camera_quaternions": [b,4]

        if self._skip_frame_example(ex, LEFT_LENS_NAME, stereo_mode):
          continue

        _, _, _, _, baseline_in_mm = self.camera_intrinsics(
            sensor_name, LEFT_LENS_NAME
        )

        sample = {
            FEATURE_SESSION_TYPE: self.type,
            FEATURE_IMAGE: str(ex.rgb_filename(LEFT_LENS_NAME)),
            FEATURE_CAMERA_BASELINE_IN_METERS: baseline_in_mm / 1000.0,
        }

        included_camera_metadata_lenses = [LEFT_LENS_NAME]

        if stereo_mode:
          sample[FEATURE_IMAGE_RIGHT] = str(ex.rgb_filename(RIGHT_LENS_NAME))
          included_camera_metadata_lenses.append(RIGHT_LENS_NAME)

        for lens_name in included_camera_metadata_lenses:
          is_left_lens = lens_name == LEFT_LENS_NAME
          fx_ratio, fy_ratio, cx_ratio, cy_ratio, _ = self.camera_intrinsics(
              sensor_name, lens_name
          )

          sample[
              FEATURE_CAMERA_INTRINSICS
              if is_left_lens
              else FEATURE_CAMERA_RIGHT_INTRINSICS
          ] = [
              fx_ratio,
              fy_ratio,
              cx_ratio,
              cy_ratio,
          ]

        if self.config.feature_metric_depth.to_include():
          metric_depth_filename = ex.metric_depth_filename(LEFT_LENS_NAME)
          sample[FEATURE_METRIC_DEPTH_LABEL] = str(metric_depth_filename)
          sample[FEATURE_HAS_METRIC_DEPTH_LABEL] = (
              metric_depth_filename.exists()
          )

        if self.config.feature_panoptic_mask.to_include():
          segmentation_mask_filename = ex.segmentation_mask_filename(
              LEFT_LENS_NAME
          )
          sample[FEATURE_PANOPTIC_MASK_LABEL] = str(segmentation_mask_filename)
          sample[FEATURE_HAS_PANOPTIC_MASK_LABEL] = (
              segmentation_mask_filename.exists()
          )

        if self.config.feature_metric_depth_zed.to_include():
          metric_zed_depth_filename = ex.metric_depth_zed_filename(
              LEFT_LENS_NAME
          )
          sample[FEATURE_METRIC_DEPTH_ZED_LABEL] = str(
              metric_zed_depth_filename
          )
          sample[FEATURE_HAS_METRIC_DEPTH_ZED_LABEL] = (
              metric_zed_depth_filename.exists()
          )

        if self.config.feature_camera_pose.to_include():
          sample[FEATURE_TRACKING_STATE] = self.camera_poses(sensor_name)[
              frame_num
          ][FEATURE_TRACKING_STATE]
          sample[FEATURE_CAMERA_TRANSLATIONS] = self.camera_poses(sensor_name)[
              frame_num
          ][FEATURE_CAMERA_TRANSLATIONS]
          sample[FEATURE_CAMERA_QUATERNIONS] = self.camera_poses(sensor_name)[
              frame_num
          ][FEATURE_CAMERA_QUATERNIONS]

        yield sample

  def itersamples(
      self,
  ) -> Iterator[Mapping[str, Any]]:
    """Yield sample dictionary based on the view and config."""
    if self.config.dataset_view_mode not in [
        DatasetViewMode.MONO_VIEW_FRAME_MODE,
        DatasetViewMode.STEREO_VIEW_FRAME_MODE,
    ]:
      raise ValueError(
          '`%s` not currently supported'
          % self.config.dataset_view_mode.value.upper()
      )

    # TODO(sagarwaghmare, kwilber): Add support for other DatasetViewMode modes.

    return self.itersamples_frame_mode(
        stereo_mode=self.config.dataset_view_mode
        == DatasetViewMode.STEREO_VIEW_FRAME_MODE
    )


@dataclasses.dataclass
class FrameExample:
  """A single frame from SANPO, coming from a single sensor."""

  session: SanpoSession
  sensor_name: str
  frame_num: int

  @classmethod
  def from_session(
      cls, session, sensor_name, frame_num
  ) -> Optional['FrameExample']:
    ex = FrameExample(session, sensor_name, frame_num)
    if ex.rgb_filename(LEFT_LENS_NAME).exists():
      return ex

  @property
  def has_right_lens(self):
    return self.rgb_filename(RIGHT_LENS_NAME).exists()

  @property
  def has_metric_depth(self):
    return self.metric_depth_filename(LEFT_LENS_NAME).exists()

  @property
  def has_metric_depth_zed(self):
    return self.metric_depth_zed_filename(LEFT_LENS_NAME).exists()

  @property
  def has_segmentation_mask(self):
    return self.segmentation_mask_filename(LEFT_LENS_NAME).exists()

  @property
  def has_camera_pose(self):
    # all examples have some camera pose
    return True

  def rgb_filename(self, lens_name: str) -> pathlib.Path:
    return (
        self.session.base_path
        / self.sensor_name
        / lens_name
        / RGB_FRAME_DIRNAME
        / FILENAME_RGB.format(frame_num=self.frame_num)
    )

  def segmentation_mask_filename(self, lens_name: str) -> pathlib.Path:
    return (
        self.session.base_path
        / self.sensor_name
        / lens_name
        / SEGMENTATION_MASK_DIRNAME
        / FILENAME_SEMANTIC.format(frame_num=self.frame_num)
    )

  def metric_depth_filename(self, lens_name: str) -> pathlib.Path:
    return (
        self.session.base_path
        / self.sensor_name
        / lens_name
        / DEPTH_DIRNAME
        / FILENAME_DEPTH.format(frame_num=self.frame_num)
    )

  def metric_depth_zed_filename(self, lens_name: str) -> pathlib.Path:
    return (
        self.session.base_path
        / self.sensor_name
        / lens_name
        / ZED_DEPTH_DIRNAME
        / FILENAME_DEPTH.format(frame_num=self.frame_num)
    )


def get_labelmap_name_to_id(folder=None, filename='labelmap_open_source.json'):
  """Get SANPO label map, mapping names to IDs.

  Args:
    folder: Path containing the label map
    filename: Basename of label map

  Returns:
    A mapping from SANPO label name to semantic ID.
  """
  if folder is None:
    folder = wrapped_path(__file__).parent
  labelmap_filename = wrapped_path(folder) / filename
  return json.loads(labelmap_filename.read_text())


def get_labelmap_id_to_name(folder=None, filename='labelmap_open_source.json'):
  name_to_id = get_labelmap_name_to_id(folder, filename)
  return {label_id: name for name, label_id in name_to_id.items()}
