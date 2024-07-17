import os
from tqdm import tqdm


DATASET_DIR = '../video_data/VCDB/distractors/videos'
OUTPUT_DIR = '../video_data/VCDB/distractors/frames'

FPS = 1


def extract_frames(output_dir, dataset_dir, fps):
    '''
    -loglevel quiet : do not output logs in the cmd
    -y : overwrite original image file if existed
    -start_number 0 : name frame from 0
    -q x : keep original quality of images
    -i {} : original video path
    -vf fps=x {} : save frames every 1/fps second(s) to the output path
    '''
    cmd_str = 'ffmpeg -loglevel quiet -y -start_number 0 -q 0 -vf fps={} {} -i {}' 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_folder_lst = os.listdir(dataset_dir)

    for video_folder in tqdm(video_folder_lst):
        video_dir = os.path.join(dataset_dir, video_folder)
        video_lst = os.listdir(video_dir)
        for video in video_lst:
            video_path = os.path.join(video_dir, video)
            video_name = video.split('.')[0]
            # frame_dir = os.path.join(output_dir, video_folder, video_name)
            frame_dir = os.path.join(output_dir, video_name)
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            frame_path = os.path.join(frame_dir, '%05d.jpg')
            cmd = cmd_str.format(fps, frame_path, video_path)
            os.popen(cmd)


def checkout_frames(output_dir):
    video_folder_lst = os.listdir(output_dir)
    print(len(video_folder_lst))


if __name__ == '__main__':
    extract_frames(output_dir=OUTPUT_DIR, dataset_dir=DATASET_DIR, fps=FPS)
    checkout_frames(output_dir=OUTPUT_DIR)

