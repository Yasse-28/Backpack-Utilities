import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.synchronise_images import SynchronizedData


def extract_images_from_sequence(video_filename,
                                 saving_path=None,
                                 starting_index=0):
    cap = cv2.VideoCapture(video_filename)
    print(f"Total number of frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_index)
    i = 0
    if not (saving_path is None):
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)
        names_file = open(os.path.join(saving_path.removesuffix('images'), "rgb.txt"), "wt")

    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, frame = cap.read()
        frame = frame[::2, ::2, :]
        if not (saving_path is None):
            img_filename = os.path.join(saving_path, f"image_{i:05d}.png")
            names_file.write(f"image_{i:05d}.png\n")
            cv2.imwrite(img_filename, frame)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1
        # if i > 800:
        #     break

    names_file.close()
    cap.release()
    cv2.destroyAllWindows()


def extract_images_tstamps_from_folder(sync_data, scl=1, display=False):
    for i in range(sync_data.num_cameras):
        saving_path = sync_data.get_saving_path(i)
        video_filename = sync_data.get_video_path(i)
        starting_index = sync_data.get_init_frame(i)
        num_frames = sync_data.get_num_frames()
        timestamps = sync_data.get_timestamps(i)
        cap = cv2.VideoCapture(video_filename)
        print(f'Saving data from video {video_filename} to {saving_path} ...')
        print(f'Starting index: {starting_index}, number of frames: {num_frames}')
        print(f"Total number of frames in the video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, starting_index)
        if not (saving_path is None):
            if not os.path.isdir(saving_path):
                os.mkdir(saving_path)
            names_file = open(os.path.join(saving_path.removesuffix('rgb'), "rgb.txt"), "wt")

        for img_id in tqdm(range(starting_index, starting_index + num_frames)):
            _, frame = cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = frame[::scl, ::scl, :]
            h, w, *_ = frame.shape
            # frame = cv2.resize(frame, (w // scl, h // scl), interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.resize(frame, (w // scl, h // scl), interpolation=cv2.INTER_LANCZOS4)
            if not (saving_path is None):
                img_filename = os.path.join(saving_path, f"{timestamps[img_id]:10.6f}.jpg")
                names_file.write(f"{timestamps[img_id]:10.6f} rgb/{timestamps[img_id]:10.6f}.png\n")
                cv2.imwrite(img_filename, frame)

            if display:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # if i > 800:
            #     break

        names_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total number of saved frames: {num_frames}")


if __name__ == "__main__":
    # video_path = "/media/robotvision/data/datasets/MEMEX_Pilots/Lisbon/Mission_2/Sequence_3/B1/VID_20220705_165259.mp4"
    # # saving_path = "/home/robotvision/workspace_yasse/datasets/nerf/pilots/lisbon/test/images"
    # saving_path = "/media/robotvision/data/datasets/MEMEX_Pilots/Lisbon/Mission_2/Sequence_3/B1/images"
    # first_frame = 0  # 15800
    # extract_images_from_sequence(video_filename=video_path, saving_path=saving_path, starting_index=first_frame)
    base_path = '/media/robotvision/data/datasets/MEMEX_Pilots/Genoa'
    names = ['T4', 'B4','T3', 'B3']
    # names = ['B1', 'T1']
    init_index = 0
    sync_data = SynchronizedData(base_path, names)
    extract_images_tstamps_from_folder(sync_data, scl=10)
