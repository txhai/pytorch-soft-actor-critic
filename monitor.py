# Using OpenCV instead of original gym monitor for more customization
# Based on https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py and
# https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py

import cv2
import os
import numpy as np
from gym.wrappers import TimeLimit, Monitor
from gym.utils import closer
from datetime import timedelta

MONITOR_FILE_PREFIX = 'env.video'


class TimeLimitMonitor(TimeLimit):
    def __init__(self,
                 env: TimeLimit,
                 video_dir,
                 force=False,
                 video_callable=None,
                 encoder='mp4v',
                 resolution='500x500',
                 frames_per_sec=30,
                 each_video_per_ep=False):
        super(TimeLimitMonitor, self).__init__(env, env._max_episode_seconds, env._max_episode_steps)

        if force:
            clear_monitor_files(video_dir)
        elif len(detect_monitor_files(video_dir)) > 0:
            raise FileExistsError(f"Trying to write to monitor directory {video_dir} with existing monitor files. You "
                                  "should use a unique directory for each training run, or use 'force=True' to "
                                  "automatically clear previous monitor files.")

        modes = env.metadata.get('render.modes', [])
        if 'rgb_array' not in modes:
            print('Disabling video recorder because we only support video mode "rgb_array"')
            # Whoops, turns out we shouldn't be enabled after all
            self.enabled = False
            return

        if self.spec._env_name not in ['Ant', 'Hopper', 'HalfCheetah', 'Humanoid']:
            print('Disabling video recorder because we only support these environments: Ant, Hopper, HalfCheetah, Humanoid')
            self.enabled = False
            return

        self.__each_video_per_ep = each_video_per_ep
        self.episode_id = 0
        self.video_callable = self.__capture_every_episode if video_callable is None else video_callable
        self.enabled = True
        self.env_name = str(self.env).replace('<','[').replace('>',']')
        self.main_video_path = os.path.join(video_dir, f"{MONITOR_FILE_PREFIX}.{self.env_name}.total.mp4")
        self.episode_video_path = ''
        self.video_dir = video_dir
        self.metadata_path = os.path.join(video_dir, f"{MONITOR_FILE_PREFIX}.{self.env_name}.txt")
        self.resolution = tuple(int(x) for x in resolution.split('x', 2))
        self.frames_per_sec = frames_per_sec
        self.env_semantics_autoreset = env.metadata.get('semantics.autoreset')
        self.num_frames = 0
        self.encoder = encoder
        # initializing OpenCV VideoWriter
        self.video_encoder: cv2.VideoWriter = cv2.VideoWriter(self.main_video_path,
                                                              cv2.VideoWriter_fourcc(*encoder),
                                                              self.frames_per_sec,
                                                              self.resolution)
        self.episode_video_encoder: cv2.VideoWriter = None
        self._monitor_id = monitor_closer.register(self)

    def __generate_text_frame(self, text, text_color=(0, 210, 180)):
        img = np.zeros((*self.resolution, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = text_color
        line_type = 2
        text = f"{text}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = int((img.shape[1] - text_size[0]) / 2)
        text_y = int((img.shape[0] + text_size[1]) / 2)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, line_type)
        return img

    def __write_episode_frames(self, duration=2): # 2 seconds
        for i in range(self.frames_per_sec * duration):
            self.video_encoder.write(self.__generate_text_frame(f"Episode {self.episode_id}"))
            if self.episode_video_encoder:
                self.episode_video_encoder.write(self.__generate_text_frame(f"Episode {self.episode_id}"))
            self.num_frames += 1

    def __capture_every_episode(self, episode_id):
        return True

    def __is_functional(self):
        return self.video_callable(self.episode_id) and self.enabled

    def __capture_frame(self):
        if not self.__is_functional():
            return

        frame = self.env.render(mode='rgb_array')

        if frame is None:
            frame = self.__generate_text_frame('Error Frame', (0, 0, 255))

        self.video_encoder.write(cv2.resize(frame, self.resolution))
        if self.episode_video_encoder:
            self.episode_video_encoder.write(cv2.resize(frame, self.resolution))

        self.num_frames += 1

    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record video
        self.__capture_frame()

        return done

    def _flush(self):
        current_time = str(timedelta(seconds=self.num_frames / self.frames_per_sec))
        with open(self.metadata_path, 'w' if not os.path.exists(self.metadata_path) else 'a') as wd:
            wd.write(f"Episode {self.episode_id}: {current_time} ({self.episode_video_path})\n")

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)
        return observation

    def _after_reset(self, observation):
        if not self.enabled:
            return
        self.reset_video_recorder()
        # Bump *after* all reset activity has finished
        self.episode_id += 1
        self._flush()

    def close(self):
        super(TimeLimitMonitor, self).close()

        if not self.enabled:
            return
        self._close_video_recorder()
        # Stop tracking this for autoclose
        monitor_closer.unregister(self._monitor_id)
        self.enabled = False

    def reset_video_recorder(self):
        if self.episode_video_encoder:
            self.episode_video_encoder.release()
            self.episode_video_encoder = None

        if not self.__is_functional():
            return

        if self.__each_video_per_ep:
            self.episode_video_path = os.path.join(self.video_dir, f"{MONITOR_FILE_PREFIX}.{self.env_name}.episode_{self.episode_id}.mp4")
            self.episode_video_encoder = cv2.VideoWriter(self.episode_video_path,
                                                         cv2.VideoWriter_fourcc(*self.encoder),
                                                         self.frames_per_sec,
                                                         self.resolution)

        self.__write_episode_frames()
        self.__capture_frame()

    def _close_video_recorder(self):
        if self.episode_video_encoder:
            self.episode_video_encoder.release()
            self.episode_video_encoder = None
        self.video_encoder.release()

    def __del__(self):
        # Make sure we've closed up shop when garbage collecting
        self.close()


def detect_monitor_files(video_dir):
    return [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.startswith(MONITOR_FILE_PREFIX)]


def clear_monitor_files(video_dir):
    files = detect_monitor_files(video_dir)
    if len(files) == 0:
        return

    print(f'Clearing {len(files)} monitor files from previous run (because force=True was provided)')
    for file in files:
        os.unlink(file)

monitor_closer = closer.Closer()