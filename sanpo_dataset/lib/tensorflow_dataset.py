# Copyright 2023 SANPO Team.
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

"""Tensorflow data loaders for SANPO."""

import collections
import functools
import io
import itertools
import os
import pathlib
import random
from typing import Any, Iterator, Mapping, Optional, Tuple, Union

import numpy as np
from sanpo_dataset.lib import common
import tensorflow as tf


TRAIN_SPLITNAME = 'train'
TEST_SPLITNAME = 'test'

_SANPO_REAL_DIRNAME = 'sanpo-real'
_SANPO_SYNTHETIC_DIRNAME = 'sanpo-synthetic'
_TRAIN_SPLIT_FILENAME = 'splits/train_session_ids.txt'
_TEST_SPLIT_FILENAME = 'splits/test_session_ids.txt'
_DATA_SPLITNAMES = [TRAIN_SPLITNAME, TEST_SPLITNAME]
_INSTANCE_ID_DIVISOR = 256.0


class SanpoDataset:
  """Dataset builder for SANPO dataset."""

  def __init__(
      self,
      dataset_path: str,
      builder_config: common.SanpoConfig,
      **builder_overrides,
  ) -> None:
    self.builder_config = builder_config.replace(**builder_overrides)
    self._dataset_path = dataset_path

    # Verify target sizes
    if self.builder_config.target_shape is not None:
      target_h, target_w = self.builder_config.target_shape
      if target_h > target_w:
        raise ValueError(
            'target_shape should be [height,width], but you set it to '
            f'[{target_h}, {target_w}] which looks like [width,height].'
        )
      if abs(target_w * 9 / 16 - target_h) > 1:
        raise ValueError(
            f'The target shape [{target_h},{target_w}] aspect ratio must be'
            f' 16:9. Consider setting a target_shape of either [{target_h},'
            f' {int(target_h*16/9)}] or [{int(target_w*9/16)}, {target_w}],'
            ' which would preserve the image aspect ratio.\n\nSANPO does not'
            ' perform cropping or color augmentation for you because'
            ' preprocessing strategies can vary by application.'
            # TODO(kwilber): add a crop tool and uncomment the below lines
            # f'To crop the image, you can use the `common.crop_*` '
            # f'family of functions which properly adjust camera intrinsics.'
        )

    # TODO(kwilber): Verify the config.
    self._data_sessions = collections.defaultdict(list)
    if self.builder_config.include_real:
      real_dataset_path = os.path.join(dataset_path, _SANPO_REAL_DIRNAME)
      self._real_sessions_train_list = common.SanpoSessionList(
          real_dataset_path,
          session_ids_or_ids_file=os.path.join(
              real_dataset_path, _TRAIN_SPLIT_FILENAME
          ),
          config=self.builder_config,
      )
      self._data_sessions[TRAIN_SPLITNAME].extend(
          self._real_sessions_train_list.get_valid_sessions()
      )
      self._real_sessions_test_list = common.SanpoSessionList(
          real_dataset_path,
          session_ids_or_ids_file=os.path.join(
              real_dataset_path, _TEST_SPLIT_FILENAME
          ),
          config=self.builder_config,
      )
      self._data_sessions[TEST_SPLITNAME].extend(
          self._real_sessions_test_list.get_valid_sessions()
      )

    if self.builder_config.include_synthetic:
      synthetic_dataset_path = os.path.join(
          dataset_path, _SANPO_SYNTHETIC_DIRNAME
      )
      self._synthetic_sessions_train_list = common.SanpoSessionList(
          synthetic_dataset_path,
          session_ids_or_ids_file=os.path.join(
              synthetic_dataset_path, _TRAIN_SPLIT_FILENAME
          ),
          config=self.builder_config,
      )
      self._data_sessions[TRAIN_SPLITNAME].extend(
          self._synthetic_sessions_train_list.get_valid_sessions()
      )
      self._synthetic_sessions_test_list = common.SanpoSessionList(
          synthetic_dataset_path,
          session_ids_or_ids_file=os.path.join(
              synthetic_dataset_path, _TEST_SPLIT_FILENAME
          ),
          config=self.builder_config,
      )
      self._data_sessions[TEST_SPLITNAME].extend(
          self._synthetic_sessions_test_list.get_valid_sessions()
      )

    if not self._data_sessions[TRAIN_SPLITNAME]:
      raise ValueError('Train split is empty.')

    if not self._data_sessions[TEST_SPLITNAME]:
      raise ValueError('Test split is empty.')

    # Shuffle the train and test sessions ids.
    for _, sessions_list in self._data_sessions.items():
      random.shuffle(sessions_list)

  def _get_tensor_signature(self) -> Mapping[str, tf.TensorSpec]:
    """Return output signature for tf.data.Dataset `from_generator`."""

    signature = {
        common.FEATURE_SESSION_TYPE: tf.TensorSpec(shape=(), dtype=tf.string),
        common.FEATURE_IMAGE: tf.TensorSpec(shape=(), dtype=tf.string),
        common.FEATURE_FRAME_ID: tf.TensorSpec(shape=(), dtype=tf.string),
        common.FEATURE_CAMERA_BASELINE_IN_METERS: tf.TensorSpec(
            shape=(), dtype=tf.float32
        ),
        common.FEATURE_CAMERA_INTRINSICS: tf.TensorSpec(
            shape=(4,), dtype=tf.float32
        ),
    }
    if self.builder_config.dataset_view_mode.is_stereo_mode():
      signature[common.FEATURE_IMAGE_RIGHT] = tf.TensorSpec(
          shape=(), dtype=tf.string
      )
      signature[common.FEATURE_CAMERA_RIGHT_INTRINSICS] = tf.TensorSpec(
          shape=(4,), dtype=tf.float32
      )

    if self.builder_config.feature_metric_depth.to_include():
      signature[common.FEATURE_METRIC_DEPTH_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.string
      )
      signature[common.FEATURE_HAS_METRIC_DEPTH_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.bool
      )

    if self.builder_config.feature_metric_depth_zed.to_include():
      signature[common.FEATURE_METRIC_DEPTH_ZED_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.string
      )
      signature[common.FEATURE_HAS_METRIC_DEPTH_ZED_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.bool
      )

    if self.builder_config.feature_panoptic_mask.to_include():
      signature[common.FEATURE_PANOPTIC_MASK_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.string
      )
      signature[common.FEATURE_HAS_PANOPTIC_MASK_LABEL] = tf.TensorSpec(
          shape=(), dtype=tf.bool
      )

    if self.builder_config.feature_camera_pose.to_include():
      signature[common.FEATURE_TRACKING_STATE] = tf.TensorSpec(
          shape=(), dtype=tf.bool
      )
      signature[common.FEATURE_CAMERA_TRANSLATIONS] = tf.TensorSpec(
          shape=(3,), dtype=tf.float32
      )
      signature[common.FEATURE_CAMERA_QUATERNIONS] = tf.TensorSpec(
          shape=(4,), dtype=tf.float32
      )

    return signature

  def _maybe_resize(
      self, tensor: tf.Tensor, *, use_nearest_neighbor: bool
  ) -> tf.Tensor:
    """Optionally resize the input tensor."""
    if self.builder_config.target_shape is None:
      return tensor
    target_h, target_w = self.builder_config.target_shape
    if use_nearest_neighbor:
      resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    else:
      resize_method = tf.image.ResizeMethod.BILINEAR
    return tf.image.resize(tensor, [target_h, target_w], method=resize_method)

  def _tf_decode_image(self, filename: tf.Tensor) -> tf.Tensor:
    # can't use tf.io.decode_image here because
    # https://github.com/tensorflow/tensorflow/issues/9356
    return tf.io.decode_png(
        tf.io.read_file(filename),
        channels=3,
        dtype=tf.uint8,
    )

  def _tf_load_image(self, filename: tf.Tensor) -> tf.Tensor:
    image = self._tf_decode_image(filename)
    return tf.image.convert_image_dtype(
        image, tf.float32, saturate=False, name=None
    )

  def _np_load_npz(self, filename: tf.Tensor) -> tf.Tensor:
    with common.wrapped_open(filename.numpy(), 'rb') as f:
      b = io.BytesIO(f.read())
      npz = np.load(b)
    assert len(npz.files) == 1
    array = np.expand_dims(npz[npz.files[0]], -1).astype(np.float32)
    return tf.convert_to_tensor(array, dtype=tf.float32)

  def _tf_load_npz(self, filename: tf.Tensor) -> tf.Tensor:
    arr = tf.py_function(self._np_load_npz, [filename], Tout=[tf.float32])[0]
    return tf.ensure_shape(arr, [None, None, 1])

  def _tf_load_float16_gzipped(self, filename: tf.Tensor) -> tf.Tensor:
    data_tensor = tf.io.decode_raw(
        tf.io.decode_compressed(tf.io.read_file(filename), 'GZIP'),
        tf.float16,
        little_endian=True,
    )
    height = data_tensor[0]
    width = data_tensor[1]
    x = tf.reshape(data_tensor[2:], [height, width, 1])
    return tf.ensure_shape(x, [None, None, 1])

  @tf.function
  def _tf_load_panoptic_labels(
      self,
      features: Mapping[str, tf.Tensor],
      image: tf.Tensor,
  ) -> Mapping[str, tf.Tensor]:
    """Loads panoptic segmentation labels if they are included."""
    if common.FEATURE_PANOPTIC_MASK_LABEL in features:
      if features[common.FEATURE_HAS_PANOPTIC_MASK_LABEL]:
        mask = tf.cast(
            self._tf_decode_image(features[common.FEATURE_PANOPTIC_MASK_LABEL]),
            dtype=tf.float32,
        )
      else:
        mask = tf.zeros_like(image, dtype=tf.float32)

      # We save panoptic label as a 3 channel image.
      # First channel contains the semantic label.
      # Instance id can be computed using the second and third channel
      # using the following formula:
      # `instance_id = mask[:,:,1] * 256 + mask[:,:,2].`
      # The panoptic label is computed using the following formula:
      # `panoptic_label = semantic_label * label_divisor + instance_id.`

      semantic_label = mask[:, :, 0]
      instance_id = (
          tf.math.scalar_mul(_INSTANCE_ID_DIVISOR, mask[:, :, 1])
          + mask[:, :, 2]
      )
      semantic_label = tf.expand_dims(semantic_label, -1)
      instance_id = tf.expand_dims(instance_id, -1)
      semantic_label = self._maybe_resize(
          semantic_label, use_nearest_neighbor=True
      )
      instance_id = self._maybe_resize(instance_id, use_nearest_neighbor=True)
      return {
          common.FEATURE_SEMANTIC_LABEL: semantic_label,
          common.FEATURE_INSTANCE_ID: instance_id,
          common.FEATURE_HAS_PANOPTIC_MASK_LABEL: tf.convert_to_tensor(
              features[common.FEATURE_HAS_PANOPTIC_MASK_LABEL]
          ),
      }

    return {}

  @tf.function
  def _tf_load_camera_pose(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Returns camere pose tensors if they are included."""
    if common.FEATURE_TRACKING_STATE in features:
      return {
          common.FEATURE_TRACKING_STATE: features[
              common.FEATURE_TRACKING_STATE
          ],
          common.FEATURE_CAMERA_TRANSLATIONS: features[
              common.FEATURE_CAMERA_TRANSLATIONS
          ],
          common.FEATURE_CAMERA_QUATERNIONS: features[
              common.FEATURE_CAMERA_QUATERNIONS
          ],
      }

    return {}

  @tf.function
  def _tf_load_depth(
      self,
      features: Mapping[str, tf.Tensor],
      image: tf.Tensor,
  ) -> Mapping[str, tf.Tensor]:
    """Returns zed depth tensors if they are included."""
    new_feats = {}
    if common.FEATURE_METRIC_DEPTH_ZED_LABEL in features:
      if features[common.FEATURE_HAS_METRIC_DEPTH_ZED_LABEL]:
        zed_depth = tf.cast(
            self._tf_load_float16_gzipped(
                features[common.FEATURE_METRIC_DEPTH_ZED_LABEL]
            ),
            tf.float32,
        )
      else:
        zed_depth = tf.zeros_like(image[:, :, 0], dtype=tf.float32)
        zed_depth = tf.expand_dims(zed_depth, -1)
      new_feats[common.FEATURE_HAS_METRIC_DEPTH_ZED_LABEL] = features[
          common.FEATURE_HAS_METRIC_DEPTH_ZED_LABEL
      ]
      zed_depth = self._maybe_resize(zed_depth, use_nearest_neighbor=True)
      new_feats[common.FEATURE_METRIC_DEPTH_ZED_LABEL] = zed_depth

    if common.FEATURE_METRIC_DEPTH_LABEL in features:
      if features[common.FEATURE_HAS_METRIC_DEPTH_LABEL]:
        metric_depth = tf.cast(
            self._tf_load_float16_gzipped(
                features[common.FEATURE_METRIC_DEPTH_LABEL]
            ),
            tf.float32,
        )
      else:
        metric_depth = tf.zeros_like(image[:, :, 0], dtype=tf.float32)
        metric_depth = tf.expand_dims(metric_depth, -1)
      new_feats[common.FEATURE_HAS_METRIC_DEPTH_LABEL] = features[
          common.FEATURE_HAS_METRIC_DEPTH_LABEL
      ]
      metric_depth = self._maybe_resize(metric_depth, use_nearest_neighbor=True)
      new_feats[common.FEATURE_METRIC_DEPTH_LABEL] = metric_depth

    return new_feats

  def _load_data_files(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Returns dictionary of loaded feature files."""

    loaded_features = {}
    for passthrough_feature in [
        common.FEATURE_SESSION_TYPE,
        common.FEATURE_FRAME_ID,
        common.FEATURE_CAMERA_BASELINE_IN_METERS,
        common.FEATURE_CAMERA_INTRINSICS,
        common.FEATURE_CAMERA_RIGHT_INTRINSICS,
    ]:
      if passthrough_feature in features:
        loaded_features[passthrough_feature] = features[passthrough_feature]

    # load images
    for image_feature in [
        common.FEATURE_IMAGE,
        common.FEATURE_IMAGE_RIGHT,
    ]:
      if image_feature in features:
        loaded_features[image_feature] = self._maybe_resize(
            self._tf_load_image(features[image_feature]),
            use_nearest_neighbor=False,
        )

    # Load panoptic segmentation labels
    loaded_features.update(
        self._tf_load_panoptic_labels(
            features, loaded_features[common.FEATURE_IMAGE]
        )
    )
    # Load camera pose.
    loaded_features.update(self._tf_load_camera_pose(features))
    # Load zed depth.
    loaded_features.update(
        self._tf_load_depth(features, loaded_features[common.FEATURE_IMAGE])
    )

    return loaded_features

  def _samples_to_tf_frame_features(
      self, samples: Iterator[Mapping[str, Any]]
  ) -> Iterator[Mapping[str, tf.Tensor]]:
    """Iterate over {feature_name: tensor, ...} frames in a session."""
    for sample in samples:
      features = {}
      for feature_name, feature_value in sample.items():
        if isinstance(feature_value, pathlib.Path):
          feature_value = feature_value.as_posix()
        features[feature_name] = tf.convert_to_tensor(feature_value)
      yield features

  def _session_to_frame_dataset(
      self, session: common.SanpoSession
  ) -> tf.data.Dataset:
    """Creates a tf.data.Dataset of individual frames from a SanpoSession."""
    return self._samples_to_frame_dataset(session.all_frame_itersamples())

  def _samples_to_frame_dataset(
      self, samples: Iterator[Mapping[str, Any]]
  ) -> tf.data.Dataset:
    """Creates a tf.data.Dataset of frames from the given samples."""
    tf_features = functools.partial(
        self._samples_to_tf_frame_features, samples=samples
    )
    return tf.data.Dataset.from_generator(
        tf_features,
        output_signature=self._get_tensor_signature(),
    )

  def _session_to_video_datasets(
      self, session: common.SanpoSession
  ) -> Iterator[tf.data.Dataset]:
    """Creates tf.data.Datasets of video clips from a SanpoSession."""
    for video_samples in session.video_itersamples():
      ds = self._samples_to_frame_dataset(video_samples)

      if self.builder_config.video_frame_stride:
        ds = ds.shard(self.builder_config.video_frame_stride, 0)

      if not self.builder_config.num_video_frames:
        raise ValueError('num_video_frames must be specified')

      # Batch the frames into video clips of length num_video_frames. We must
      # use drop_remainder=True to ensure each video clip has the exact same
      # number of frames because the later mapping to load the data files
      # requires rebatching. This is done so that all the video clips across
      # multiple sessions can be shuffled together before performing the
      # expensive data loading step.
      ds = ds.batch(
          self.builder_config.num_video_frames,
          drop_remainder=True,
      )

      yield ds

  def to_tf_data(
      self,
      split_name: Optional[str] = None,
  ) -> Union[Tuple[tf.data.Dataset, tf.data.Dataset], tf.data.Dataset]:
    """Creates tf.data.Datasets for train and test splits of the session.

    Args:
      split_name: If specified, returns a single dataset for this split. If
        None, a dataset for each train and test split is returned. Default: None

    Returns:
      A tuple containing the train and test datasets, or a single dataset if
      split_name was specified.
    """
    result_datasets = []
    if split_name and split_name not in _DATA_SPLITNAMES:
      raise ValueError('split_name must be one of {}'.format(_DATA_SPLITNAMES))

    split_names = [split_name] if split_name else _DATA_SPLITNAMES
    for split_name in split_names:
      sessions = self._data_sessions[split_name]

      # Convert each session into a dataset.
      if self.builder_config.dataset_view_mode.is_video_mode():
        # For video mode the dataset will contain one or more 'video clips'
        # (batch of frames).
        session_datasets = itertools.chain.from_iterable(
            map(self._session_to_video_datasets, sessions)
        )
      else:
        # For frame mode the dataset will contain one sample for each video
        # frame.
        session_datasets = map(self._session_to_frame_dataset, sessions)

      # Concat all the session datasets together.
      dataset = next(session_datasets)
      for ds in session_datasets:
        dataset = dataset.concatenate(ds)

      # Cache the dataset before shuffling.
      dataset = dataset.cache()

      # The dataset features now contain filenames. Below we shuffle the
      # data together (frames or video clips) and then load the data from files.
      # Since loading the data is expensive and uses significant memory, it's
      # much more efficient to shuffle before loading the data.
      if self.builder_config.shuffle_buffer_size > 1:
        dataset = dataset.shuffle(
            self.builder_config.shuffle_buffer_size,
            reshuffle_each_iteration=True,
        )

      # For video mode, unbatch the video clips into individual frames so we can
      # apply the mapping function to load the data files.
      if self.builder_config.dataset_view_mode.is_video_mode():
        dataset = dataset.unbatch()

      # Map the frames by loading the underlying data files into memory.
      # For video mode this is done deterministicly to preserve the order of
      # frames for rebatching back into video clips.
      map_deterministic = self.builder_config.dataset_view_mode.is_video_mode()
      dataset = dataset.map(
          self._load_data_files,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=map_deterministic,
      )

      # For video mode, rebatch the frames into videos. Since the number of
      # frames in each video is fixed, this is guaranteed that each frame will
      # return to its correct corresponding video clip.
      if self.builder_config.dataset_view_mode.is_video_mode():
        dataset = dataset.batch(self.builder_config.num_video_frames)

      result_datasets.append(dataset)

    if len(result_datasets) == 1:
      return result_datasets[0]
    else:
      return tuple(result_datasets)
