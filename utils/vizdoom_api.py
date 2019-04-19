import vizdoom as vzd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv, hsv2rgb


class VizDoom(object):

    def __init__(self, cfg):
        super(VizDoom, self).__init__()
        
        # Global vars
        self.skiprate = cfg.skiprate
        self.num_channels = cfg.num_channels
        self.resolution = cfg.resolution
        self.terminal = np.zeros([self.resolution[0], self.resolution[1], cfg.num_channels])

        # Init game, set params
        self.game = vzd.DoomGame()
        # Scenario
        self.game.set_doom_scenario_path(cfg.vizdoom_dir + "/scenarios/basic.wad")
        self.game.set_doom_map("map01")
        # Screen
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(True)
        self.game.set_sound_enabled(True)
        # Buffers
        #self.game.set_depth_buffer_enabled(True)
        #self.game.set_labels_buffer_enabled(True)
        #self.game.set_automap_buffer_enabled(True)
        # Rendering
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False) # If HUD enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(False)
        self.game.set_render_decals(False)  # Bullet holes and blood on the walls
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)  # Smoke and blood
        self.game.set_render_messages(False)  # In-game messages
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True) # Effect upon taking damage or picking up items
        # Actions
        self.game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
        # Variables included in state
        self.game.set_available_game_variables([vzd.GameVariable.AMMO2])
        # Start/end time
        self.game.set_episode_start_time(14)
        self.game.set_episode_timeout(300)
        # Reward
        self.game.set_living_reward(-1)
        self.game.set_doom_skill(5)
        # Game mode
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.init()

    def new_episode(self):
        self.game.new_episode()
    
    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def make_action(self, action):
        return self.game.make_action(action, self.skiprate)
    
    def get_state(self):
        state = None
        if self.game.get_state() is not None:
            state = self.game.get_state().screen_buffer
        return state
    
    def get_preprocessed_state(self):
        frame = self.get_state()
        if frame is None:
            return self.terminal
        # Blur, crop, resize
        frame = cv2.GaussianBlur(frame, (39,39), 0, 0)
        frame = tf.image.central_crop(frame, 0.5)
        frame = tf.image.resize(frame, self.resolution, align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
        # Kmeans clustering
        frame = rgb2hsv(frame)
        kmeans = KMeans(n_clusters=4).fit(frame.reshape((-1, 3)))
        frame = kmeans.cluster_centers_[kmeans.labels_].reshape(frame.shape)
        frame = hsv2rgb(frame).astype(np.float32)
        # Greyscale
        if self.num_channels == 1:
            frame = tf.image.rgb_to_grayscale(frame).numpy()
            #plt.imshow(frame.reshape((frame.shape[0], frame.shape[1])), cmap="gray")
            #plt.show()
        return frame
    
