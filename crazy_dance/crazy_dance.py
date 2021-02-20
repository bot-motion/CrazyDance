""" Crazyflie quadcopter 'dances' to signals from a laptop's mic

"""

import logging
import time
import argparse
import queue
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

import sounddevice as sd

import numpy as np
from scipy import fftpack

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


URI = 'radio://0/80/2M/E7E7E7E7E7'
DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5
MIN_HEIGHT = 0.3

is_deck_attached = False

logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0, 0]

amplifier = 0
vel_signal_cap = 0.1
pos_signal_cap = 0.1


q = queue.Queue()


def move_box_limit(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        body_x_cmd = 0.2
        body_y_cmd = 0.1
        max_vel = 0.2

        while (1):
            '''if position_estimate[0] > BOX_LIMIT:
                mc.start_back()
            elif position_estimate[0] < -BOX_LIMIT:
                mc.start_forward()
            '''

            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd = -max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd = max_vel
            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd = -max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd = max_vel

            mc.start_linear_motion(body_x_cmd, body_y_cmd, 0)

            time.sleep(0.1)


def dance(scf):
    global pos_signal_cap, vel_signal_cap, amplifier

    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:

        while (1):
            if amplifier > 1000:
                body_x_cmd = pos_signal_cap
                body_y_cmd = 0.1
            else:
                body_x_cmd = 0.1
                body_y_cmd = pos_signal_cap
        
            max_vel = vel_signal_cap


            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd = -max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd = max_vel
            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd = -max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd = max_vel

            if position_estimate[2] < MIN_HEIGHT:
                body_z_cmd = 0.4
            elif position_estimate[2] > MIN_HEIGHT and position_estimate[2] < 4 * MIN_HEIGHT:
                body_z_cmd = pos_signal_cap
            elif position_estimate[2] > 4 * MIN_HEIGHT:
                body_z_cmd = -0.1

            print("p_x = ", position_estimate[0], " p_y = ", position_estimate[1], " p_z = ", position_estimate[2])
            print("v_x = ", body_x_cmd, " v_y = ", body_y_cmd, " v_z = ", body_z_cmd)

            mc.start_linear_motion(body_x_cmd, body_y_cmd, body_z_cmd)

            time.sleep(0.2)


def move_linear_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)
        mc.turn_left(180)
        time.sleep(1)
        mc.forward(0.5)
        time.sleep(1)


def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()


def log_pos_callback(timestamp, data, logconf):
    global position_estimate, pos_signal_cap, vel_signal_cap, amplifier

    # print(data, "  p =", pos_signal_cap, "  v=", vel_signal_cap)

    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']
    position_estimate[2] = data['stateEstimate.z']


def param_deck_flow(name, value_str):
    value = int(value_str)
    print(value)
    global is_deck_attached
    if value:
        is_deck_attached = True
        print('Deck is attached!')
    else:
        is_deck_attached = False
        print('Deck is NOT attached!')


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)

    global position_estimate, pos_signal_cap, vel_signal_cap, amplifier

    downsample_rate = 1
    mapping = [0]
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::downsample_rate, mapping])

    # process music to generate new motion command
    try:
        data = q.get_nowait()
    except queue.Empty:
        return

    f_sample = 1
    max_vel = 0.4
    min_vel = 0.1
    max_pos = 0.4
    min_pos = -0.4

    X = fftpack.fft(data)
    freqs = fftpack.fftfreq(len(data)) * f_sample

    x_real = np.real(X)
    amplifier = 1/abs(max(x_real)-min(x_real))
    vel_signal = amplifier * np.std(x_real)
    vel_signal_cap = min(max(vel_signal,min_vel),max_vel)

    pos_signal = amplifier * np.mean(abs(x_real))
    pos_signal_cap = min(max(pos_signal,min_pos),max_pos)
        
    # print("signal = ", amplifier,vel_signal_cap,pos_signal_cap)



if __name__ == '__main__':

    device = None
    device_info = sd.query_devices(device, 'input')
    samplerate = device_info['default_samplerate']

    stream = sd.InputStream(
        device=device, channels=1,
        samplerate=samplerate, callback=audio_callback)


    cflib.crtp.init_drivers(enable_debug_driver=False)
    with stream:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

            scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                             cb=param_deck_flow)
            time.sleep(3)

            logconf = LogConfig(name='Position', period_in_ms=10)
            logconf.add_variable('stateEstimate.x', 'float')
            logconf.add_variable('stateEstimate.y', 'float')
            logconf.add_variable('stateEstimate.z', 'float')

            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(log_pos_callback)

            if is_deck_attached:
                logconf.start()

                # move_box_limit(scf)
                # take_off_simple(scf)
                # move_linear_simple(scf)

                dance(scf)

                logconf.stop()
            else:
                print("Deck not attached")