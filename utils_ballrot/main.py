import os.path

import numpy as np
from matplotlib import pyplot as plt
import pexpect
import cv2


def integrate_2d_pos(vel_forward, vel_side, vel_turn, initial_heading=0, start_pos=(0, 0), steps=4):
    step_mag = np.sqrt(vel_forward ** 2 + vel_side ** 2) + 1e-14
    step_dir = np.arctan2(vel_side, vel_forward)
    masks = np.where(step_dir < 0)
    step_dir[masks] += 2 * np.pi
    intx = np.cumsum(vel_forward)
    inty = np.cumsum(vel_side)
    heading = initial_heading + np.cumsum(-1 * vel_turn)
    mask = np.where(heading < 0)
    while len(mask[0]) > 0:
        heading[mask] += 2 * np.pi
        mask = np.where(heading < 0)
    mask = np.where(heading > 2 * np.pi)
    while len(mask[0]) > 0:
        heading[mask] -= 2 * np.pi
        mask = np.where(heading > 2 * np.pi)
    ang_dist = np.cumsum(np.abs(vel_turn))

    step = step_mag / steps
    heading_step = np.diff(np.concatenate(((initial_heading,), heading)))
    
    mask = np.where(heading_step > np.pi)
    while len(mask[0]) > 0:
        heading_step[mask] -= 2 * np.pi
        mask = np.where(heading_step > np.pi)
    mask = np.where(heading_step < -np.pi)
    while len(mask[0]) > 0:
        heading_step[mask] += 2 * np.pi
        mask = np.where(heading_step < -np.pi)

    heading_step /= steps
    
    delta_posx = np.zeros(len(heading_step))
    delta_posy = np.zeros(len(heading_step))

    normalized_vel_forward = vel_forward / step_mag
    normalized_vel_side = vel_side / step_mag
    theta = heading + heading_step / 2
    v0 = np.cos(theta) * normalized_vel_forward + np.sin(theta) * normalized_vel_side
    v1 = -np.sin(theta) * normalized_vel_forward + np.cos(theta) * normalized_vel_side
    for i in range(steps):
        delta_posx += step * v0 
        delta_posy += step * v1
        v0 = np.cos(heading_step) * v0 - np.sin(heading_step) * v1
        v1 = np.sin(heading_step) * v0 + np.cos(heading_step) * v1
    posx = np.cumsum(delta_posx) + start_pos[0]
    posy = np.cumsum(delta_posy) + start_pos[1]
    return posx, posy


def plot_delta_rot(x, forward, side, turn, labels):
    if not isinstance(forward, tuple):
        forward = (forward,)
    if not isinstance(side, tuple):
        side = (side,)
    if not isinstance(turn, tuple):
        turn = (turn,)

    fig, axes = plt.subplots(3, 1, sharex=True)
    for i in range(len(forward)):
        axes[0].plot(x, forward[i], label=labels[i], linewidth=0.2)
        axes[1].plot(x, side[i], label=labels[i], linewidth=0.2)
        axes[2].plot(x, turn[i], label=labels[i], linewidth=0.2)
    axes[0].set_ylabel("Forward [mm/s]")
    axes[1].set_ylabel("Side [mm/s]")
    axes[2].set_ylabel("Truning [degrees/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()

    return fig


def load_fictrac(path, ball_radius=5, fps=100, columns=(5, 7, 6), inversions=(-1, -1, -1), skip_integration=False):
    """
    columns are the columns in the .dat output of fictrac
    corresponding for forward rotation, side rotation, and turning
    the default is for c2a_r=(0, 0, 0).
    inversions can be used to invert the rotation direction
    """
    dat_table = np.genfromtxt(path, delimiter=",")
    data = {}
    
    data["delta_rot_forward"] = dat_table[:, columns[0]] * ball_radius * fps * inversions[0]
    data["delta_rot_side"] = dat_table[:, columns[1]] * ball_radius * fps * inversions[0]
    data["delta_rot_turn"] = dat_table[:, columns[2]] / 2 / np.pi * 360 * fps * -1

    if not skip_integration:
        x, y = integrate_2d_pos(dat_table[:, columns[0]] * inversions[0], dat_table[:, columns[1]] * inversions[1], dat_table[:, columns[2]] * inversions[2])
        data["x"] = x * ball_radius
        data["y"] = y * ball_radius
        data["integrated_forward"] = np.cumsum(data["delta_rot_forward"]) 
        data["integrated_side"] = np.cumsum(data["delta_rot_side"]) 
    
    return data


def get_ball_parameters(img, output_dir=None):
    img = cv2.medianBlur(img, 5)
    canny_params = dict(threshold1 = 40, threshold2 = 50)
    edges = cv2.Canny(img, **canny_params)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300, param1=120, param2=10, minRadius=200, maxRadius=300)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        inside = np.inf
        x_min, y_min, r_min = np.nan, np.nan, np.nan
        for x, y, r in circles:
            if x + r > img.shape[1] or x - r < 0:
                continue
            xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            xx = xx - x
            yy = yy - y
            rr = np.sqrt(xx ** 2 + yy ** 2)
            mask = (rr < r)
            current_inside = np.diff(np.quantile(edges[mask], [0.05, 0.95]))
            if  current_inside < inside:
                x_min, y_min, r_min = x, y, r
                inside = current_inside
        if output_dir is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y, r in circles:
                cv2.circle(img, (x, y), r, (255, 255, 255), 1)
                cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (255, 128, 255), -1)
            cv2.circle(img, (x_min, y_min), r_min, (255, 0, 0), 1)
            cv2.rectangle(img, (x_min - 5, y_min - 5), (x_min + 5, y_min + 5), (255, 128, 255), -1)
            cv2.imwrite(os.path.join(output_dir, "circ_fit.jpg"), img)
        return x_min, y_min, r_min
    
    
def get_mean_image(video_file, skip_existing=True):
    directory = os.path.dirname(video_file)
    mean_frame_file = os.path.join(directory, "camera_3_mean.jpg")
    if skip_existing and os.path.isfile(mean_frame_file):
        print(f"{mean_frame_file} exists loading image from file without recomputing.")
        mean_frame = cv2.imread(mean_frame_file)[:, :, 0]
    else:
        f = cv2.VideoCapture(video_file)
        rval, frame = f.read()
        # Convert rgb to grey scale
        mean_frame = np.zeros_like(frame[:, :, 0], dtype=np.int64)
        count = 0
        while rval:
            mean_frame =  mean_frame + frame[:, :, 0]
            rval, frame = f.read()
            count += 1
        f.release()
        mean_frame = mean_frame / count
        mean_frame = mean_frame.astype(np.uint8)
        cv2.imwrite(mean_frame_file, mean_frame)
    return mean_frame


def get_circ_points_for_config(x, y, r, img_shape, n=12):
    # Compute angular limit given by image size
    theta1 = np.arcsin((img_shape[0] - y) / r)
    theta2 = 1.5 * np.pi - (theta1 - 1.5 * np.pi)

    points = []
    for theta in np.linspace(theta1, theta2, n):
        point_x = x - np.cos(theta) * r
        point_y = y - np.sin(theta) * r
        points.append(int(point_x))
        points.append(int(point_y))

    return points


def _format_list(l):
    s = repr(l)
    s = s.replace("[", "{ ")
    s = s.replace("]", " }")
    return s


def write_config_file(video_file, roi_circ, vfov=3.05, q_factor=40, c2a_src="c2a_cnrs_xz", do_display="n", c2a_t=[0, 0, 0], c2a_r=[0, 0, 0], c2a_cnrs_xz=[422, 0, 422, 0, 422, 10, 422, 10], roi_ignr=[], overwrite=False):
    directory = os.path.dirname(video_file)
    config_file = os.path.join(directory, "config.txt")
    if not overwrite and os.path.isfile(config_file):
        print(f"Not writing to {config_file} because it exists.")
        return config_file

    content = f"vfov             : {vfov:.2f}"
    content += f"\nsrc_fn           : {video_file}"
    content += f"\nq_factor         : {int(q_factor)}"
    content += f"\nc2a_src          : {c2a_src}"
    content += f"\ndo_display       : {do_display}"
    content += f"\nroi_ignr         : {_format_list(roi_ignr)}"
    content += f"\nc2a_t            : {_format_list(c2a_t)}"
    content += f"\nc2a_r            : {_format_list(c2a_r)}"
    content += f"\nc2a_cnrs_xz      : {_format_list(c2a_cnrs_xz)}"
    content += f"\nroi_circ         : {_format_list(roi_circ)}"
    
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(content)

    return config_file


def run_fictrac_config_gui(config_file, fictrac_config_gui="/home/aymanns/fictrac/bin/configGui"):
    directory = os.path.dirname(config_file)
    analyzer = pexpect.spawn(f'/bin/bash -c "cd {directory} && xvfb-run -a {fictrac_config_gui} {config_file}"', encoding="utf-8")
    analyzer.expect("\? ")
    analyzer.sendline("y")
    analyzer.expect("\? ")
    analyzer.sendline("y")
    analyzer.expect("\? ")
    analyzer.sendline("y")
    analyzer.expect('Hit ENTER to exit..')
    analyzer.sendline("")
