import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os, inspect
import textwrap
from tqdm import tqdm
from args import ArgumentParser

args = ArgumentParser().parse_args()

# kpts directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
kpts_dir = currentdir + '/pose_est/output/'+ args.video + '/input_3D/keypoints3D.npz'

def get_rot_matrix(axis, angle):
    
    axis /= np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    
    rotation_matrix = np.array([
        [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2]
    ])
    
    return rotation_matrix

def motion_retarget(vals):
    
    # retarget the base
    vals = np.array([vals[idx] - vals[1] for idx in range(vals.shape[0])])
    vec_base = vals[1, :] - vals[0, :]
    rotation_axis = np.cross(np.array([0, 0, 1]), vec_base)
    dot_product = np.dot(vec_base, np.array([0, 0, 1]))
    norm_product = np.linalg.norm(vec_base) * np.linalg.norm(np.array([0, 0, 1]))
    angle = np.arccos(dot_product / norm_product)
    rotation_matrix = get_rot_matrix(rotation_axis, angle)
    vals[0] = np.dot(rotation_matrix, vals[0].T).T
    vals = np.array([vals[idx] - vals[0] for idx in range(vals.shape[0])])

    # match the length auto-adjusting action space
    scale_matrix = [0.34, 0.4, 0.4]
    for idx in range(vals.shape[0] - 1):
        vec = (vals[idx + 1, :] - vals[idx, :])
        norm_vec = vec / np.linalg.norm(np.array(vec))
        vec_scaled = scale_matrix[idx] * norm_vec
        vals[idx + 1, :] = vec_scaled + vals[idx, :]
    
    return vals

def generate_motion(kpts):

    info_str = """
    ———————————————————————————————————————————————————————————————
    Generating Expert Arm Motion...
    ———————————————————————————————————————————————————————————————
    """
    info_str = textwrap.dedent(info_str)
    print(info_str)

    pose_out_3D = np.array([])

    for i in tqdm(range(kpts.shape[0])):
    
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')
        ax.view_init(elev=15., azim=70)
        vals = kpts[i]

        # retarget the motion from human mophology to robot's
        vals = motion_retarget(vals)

        pose_out_3D = np.append(pose_out_3D, vals)

        # plot the demo video
        I = np.array([0, 1, 2])
        J = np.array([1, 2, 3])
        for j in range(I.shape[0]):
            x, y, z = [np.array([vals[I[j], k], vals[J[j], k]]) for k in range(3)]
            ax.plot(x, y, z, lw=2)
            ax.scatter(x, y, z)
    
        RADIUS = 0.6
        SCALE_X = 1.0
        SCALE_Y = 1.0
        SCALE_Z = 1.0

        ax.set_xlim3d([-RADIUS * SCALE_X, RADIUS * SCALE_X])
        ax.set_ylim3d([-RADIUS * SCALE_Y, RADIUS * SCALE_Y])
        ax.set_zlim3d([-RADIUS * SCALE_Z, RADIUS * SCALE_Z])
        ax.set_aspect('auto')

        white = (1.0, 1.0, 1.0, 0.0)
        ax.xaxis.set_pane_color(white) 
        ax.yaxis.set_pane_color(white)
        ax.zaxis.set_pane_color(white)

        output_dir_3D_img = currentdir + '/demo_data/' + args.video +'/pose3D/'
        os.makedirs(output_dir_3D_img, exist_ok=True)
        plt.savefig(output_dir_3D_img + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
    
        plt.savefig('temp_frame.png')
        frame = cv2.imread('temp_frame.png')
        frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)
        plt.close()

    video_writer.release()
    os.remove('temp_frame.png')

    output_dir_3D_kpts = currentdir + '/demo_data/' + args.video + '/'
    os.makedirs(output_dir_3D_kpts, exist_ok=True)

    frame = int(pose_out_3D.shape[0] / (4 * 3))

    pose_out_3D = pose_out_3D.reshape(frame, 4, 3)
    output_npz = output_dir_3D_kpts + 'keypoints3D.npz'
    np.savez_compressed(output_npz, reconstruction=pose_out_3D)

    info_str = """
    ———————————————————————————————————————————————————————————————
    Expert Arm Motion Generated.
    ———————————————————————————————————————————————————————————————
    """
    info_str = textwrap.dedent(info_str)
    print(info_str)

if __name__ == '__main__':
    
    keypoints = np.load(kpts_dir, allow_pickle=True)['reconstruction']
    kpts = keypoints[:, [8, 14, 15, 16], :]

    width, height = 800, 600
    fps = 30
    output_dir_3D = currentdir + '/demo_data/' + args.video + '/arm_motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_dir_3D, fourcc, fps, (width, height))

    generate_motion(kpts)
