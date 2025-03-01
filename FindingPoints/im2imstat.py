import os
import numpy as np
import cv2
import time
import getpoint


def get_image_file_list(directory, cam_id, imnames_pattern, imgext):
    # Get list of image files matching the pattern
    search_pattern = os.path.join(directory, str(cam_id), imnames_pattern + imgext)
    return [f for f in os.listdir(search_pattern) if f.endswith(imgext)]


def compute_average_image(seq, step, img_dir):
    print(f'Computing average image for camera {seq["camId"]}')
    images = []
    pointsIdx = list(range(0, len(seq['data']), step))
    
    for idx in pointsIdx:
        img_path = os.path.join(img_dir, seq['data'][idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Adjust to load image properly
        images.append(img.astype(np.float64))
    
    avIM = np.mean(images, axis=0).astype(np.uint8)
    return avIM


def compute_std_image(seq, step, avIM, img_dir):
    print(f'Computing standard deviation image for camera {seq["camId"]}')
    images = []
    pointsIdx = list(range(0, len(seq['data']), step))
    
    for idx in pointsIdx:
        img_path = os.path.join(img_dir, seq['data'][idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Adjust to load image properly
        images.append((img.astype(np.float64) - avIM) ** 2)
    
    stdIM = np.sqrt(np.mean(images, axis=0)).astype(np.uint8)
    return stdIM


def im2imstat(config, CamIds, STEP4STAT=5):
    img_dir = config['paths']['img']
    imgext = config['files']['imgext']
    
    NoCams = len(CamIds)
    
    # Load image names
    seq = []
    for cam_id in CamIds:
        data_files = get_image_file_list(img_dir, cam_id, config['files']['imnames'], imgext)
        if len(data_files) < 4:
            raise ValueError('Not enough images found. Wrong image path or name pattern?')
        seq.append({'camId': cam_id, 'data': data_files, 'size': len(data_files)})
    
    NoPoints = min([s['size'] for s in seq])
    pointsIdx = range(0, NoPoints, STEP4STAT)
    
    # Compute average images
    start_time = time.time()
    for s in seq:
        avIM_file = config['files']['avIM'].format(s['camId'])
        if not os.path.exists(avIM_file):
            avIM = compute_average_image(s, STEP4STAT, img_dir)
            cv2.imwrite(avIM_file, avIM)
        else:
            print(f'Average image for camera {s["camId"]} already exists.')
    print(f'Elapsed time for average images: {time.time() - start_time} [sec]')
    
    # Compute standard deviation images
    start_time = time.time()
    for s in seq:
        stdIM_file = config['files']['stdIM'].format(s['camId'])
        if not os.path.exists(stdIM_file):
            avIM = cv2.imread(config['files']['avIM'].format(s['camId']), cv2.IMREAD_GRAYSCALE)
            stdIM = compute_std_image(s, STEP4STAT, avIM, img_dir)
            cv2.imwrite(stdIM_file, stdIM)
        else:
            print(f'Standard deviation image for camera {s["camId"]} already exists.')
    print(f'Elapsed time for standard deviation images: {time.time() - start_time} [sec]')
    
    # Find points in the images
    Ws, Res, IdMat = [], [], np.ones((NoCams, NoPoints), dtype=int)
    for s in seq:
        Points = []
        avIM = cv2.imread(config['files']['avIM'].format(s['camId']), cv2.IMREAD_GRAYSCALE)
        stdIM = cv2.imread(config['files']['stdIM'].format(s['camId']), cv2.IMREAD_GRAYSCALE)
        
        for j in range(NoPoints):
            img_path = os.path.join(img_dir, s['data'][j])
            pos, err = getpoint(img_path, 0, config['imgs'], avIM, stdIM)
            
            if err:
                IdMat[CamIds.index(s['camId']), j] = 0
                Points.append([np.nan, np.nan, np.nan])
            else:
                Points.append([pos[0], pos[1], 1])
        
        Ws.append(np.array(Points).T)
        Res.append([avIM.shape[1], avIM.shape[0]])  # (Width, Height)
    
    # Save results
    idx = ''.join([f"{i:02d}" for i in CamIds])
    np.savetxt(config['files']['points'] + idx + '.txt', np.vstack(Ws), fmt='%.6f')
    np.savetxt(config['files']['Res'] + idx + '.txt', np.array(Res), fmt='%d')
    np.savetxt(config['files']['IdMat'] + idx + '.txt', IdMat, fmt='%d')
    
    # Mark process as done
    donefile = config['files']['done']
    with open(donefile, 'w') as f:
        f.write('1')

    print("Process complete.")

"""
if __name__ == "__main__":
    config = read_configuration()
    CamIds = [0, 1, 2]  # Example camera IDs
    im2imstat(config, CamIds)
"""