import os
import numpy as np
import cv2
import time
import getpoint


# Placeholder function to simulate loading configuration
def read_configuration():
    config = {
        'paths': {'img': './images/'},
        'files': {
            'imgext': 'png',   # Change to the actual image extension you're using
            'idxcams': [1, 2, 3],  # Camera IDs
            'imnames': 'image_%d_',  # Naming pattern for images (image_ID_*)
            'avIM': './average_cam_%d.png',  # Path for average image files
            'stdIM': './std_cam_%d.png',  # Path for standard deviation image files
            'points': './points.txt',
            'Res': './resolution.txt',
            'IdMat': './idmat.txt',
            'maxid': 100  # Max ID for frames
        },
        'expname': 'experiment'
    }
    return config


# Main script starts here
SHOWFIG = 0  # Show images during point extraction (0 = off)
STEP4STAT = 1  # Step for computing average and std images, if 1 then all images taken

config = read_configuration()

im_dir = config['paths']['img']
im_ext = config['files']['imgext']
NoCams = len(config['files']['idxcams'])  # Number of cameras

# Load image names and verify if enough images are available
seq = []
for cam_id in config['files']['idxcams']:
    camera_data = {
        'camId': cam_id,
        'data': [],
    }
    image_files = [f for f in os.listdir(im_dir) if f.startswith(config['files']['imnames'] % cam_id) and f.endswith(im_ext)]
    camera_data['data'] = sorted(image_files)
    camera_data['size'] = len(image_files)
    
    if camera_data['size'] < 4:
        raise ValueError("Not enough images found. Check the image path or name pattern.")
    
    seq.append(camera_data)

# Create the occupancy matrix for image frames
if 'maxid' in config['files']:
    NoPoints = config['files']['maxid']
    FrameMat = np.zeros((config['files']['maxid'], NoCams), dtype=int)
    for i, camera in enumerate(seq):
        camera['imgidx'] = np.zeros(NoPoints)
        for j, img in enumerate(camera['data']):
            FrameMat[int(img[config['files']['imnames'].index('%d') + 1:]), i] = j + 1
else:
    NoPoints = min([camera['size'] for camera in seq])
    FrameMat = np.zeros((NoPoints, NoCams), dtype=int)
    for i in range(NoCams):
        FrameMat[:, i] = np.arange(1, NoPoints + 1)

# Compute average and standard deviation images
start_time = time.time()
for i, camera in enumerate(seq):
    av_im_path = config['files']['avIM'] % camera['camId']
    if not os.path.exists(av_im_path):
        print(f"Computing average image for camera {camera['camId']}")
        avIM = np.zeros(cv2.imread(os.path.join(im_dir, camera['data'][0])).shape, dtype=np.float64)
        point_idx = np.arange(0, camera['size'], STEP4STAT)
        for idx in point_idx:
            img = cv2.imread(os.path.join(im_dir, camera['data'][idx]))
            avIM += img.astype(np.float64)
        avIM = np.uint8(np.round(avIM / len(point_idx)))
        cv2.imwrite(av_im_path, avIM)
    else:
        print(f"Average image for camera {camera['camId']} already exists")

print(f"Elapsed time for computation of average images: {time.time() - start_time:.2f} seconds")

# Compute standard deviation images
start_time = time.time()
for i, camera in enumerate(seq):
    std_im_path = config['files']['stdIM'] % camera['camId']
    if not os.path.exists(std_im_path):
        avIM = cv2.imread(config['files']['avIM'] % camera['camId'], cv2.IMREAD_GRAYSCALE).astype(np.float64)
        print(f"Computing standard deviation image for camera {camera['camId']}")
        stdIM = np.zeros(avIM.shape, dtype=np.float64)
        point_idx = np.arange(0, camera['size'], STEP4STAT)
        for idx in point_idx:
            img = cv2.imread(os.path.join(im_dir, camera['data'][idx]), cv2.IMREAD_GRAYSCALE)
            stdIM += (img.astype(np.float64) - avIM) ** 2
        stdIM = np.uint8(np.round(np.sqrt(stdIM / (len(point_idx) - 1))))
        cv2.imwrite(std_im_path, stdIM)
    else:
        print(f"Standard deviation image for camera {camera['camId']} already exists")

print(f"Elapsed time for computation of standard deviation images: {time.time() - start_time:.2f} seconds")

# Find points in the images
Ws = []
Res = []
IdMat = np.ones((NoCams, NoPoints), dtype=int)

print('*********************************************')
print(f'Finding points (laser projections) in cameras: {NoCams} cameras, {NoPoints} images for each camera')
print('*********************************************')

for i, camera in enumerate(seq):
    t1 = time.time()
    print(f'Finding points in camera No: {camera["camId"]}')
    Points = []
    avIM = cv2.imread(config['files']['avIM'] % camera['camId'], cv2.IMREAD_GRAYSCALE)
    stdIM = cv2.imread(config['files']['stdIM'] % camera['camId'], cv2.IMREAD_GRAYSCALE)
    
    for j in range(NoPoints):
        print(f'Processing frame {j+1} of {NoPoints}', end="\r")
        idx2data = FrameMat[j, i]
        if idx2data:
            pos, err = getpoint(os.path.join(im_dir, camera['data'][idx2data - 1]), SHOWFIG, config['files'], avIM, stdIM)
        else:
            err = True
        
        if err:
            IdMat[i, j] = 0
            Points.append([np.nan, np.nan, np.nan])
        else:
            Points.append([pos[0], pos[1], 1])
    
    Ws.append(Points)
    Res.append([avIM.shape[1], avIM.shape[0]])
    
    t2 = time.time()
    print(f"\nElapsed time for finding points in one camera: {int((t2-t1) // 60)} minutes {int((t2-t1) % 60)} seconds")
    print(f"{np.sum([p[2] for p in Points if not np.isnan(p[2])])} points found in camera No: {camera['camId']}")

# Save the results
Ws = np.array(Ws)
Res = np.array(Res)
IdMat = np.array(IdMat)

np.savetxt(config['files']['points'], Ws.reshape(-1, 3), fmt='%f')
np.savetxt(config['files']['Res'], Res, fmt='%d')
np.savetxt(config['files']['IdMat'], IdMat, fmt='%d')

# Display overall statistics
print('Overall statistics from im2points: ************************')
print(f'Total number of frames (possible 3D points): {NoPoints}')
print(f'Total number of cameras: {NoCams}')
print('More important statistics: *******************************')
print(f'Detected 3D points: {np.sum(np.sum(IdMat, axis=0) > 0)}')
print(f'Detected 3D points in at least 3 cams: {np.sum(np.sum(IdMat, axis=0) > 2)}')
print(f'Detected 3D points in ALL cameras: {np.sum(np.sum(IdMat, axis=0) == NoCams)}')
