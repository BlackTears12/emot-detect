import glob
import subprocess
import os

src = r'dataset/CREMA-D/VideoFlash'
dst = r'dataset/CREMA-D/Extracted'

desired_frame_dim = 224


def delete_files_in_directory(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


ffmpeg_downscale_args = "scale=360:270"
ffmpeg_crop_to_square_args = "crop={desired_frame_dim}:{desired_frame_dim}:(iw-{desired_frame_dim})/2:(ih-{desired_frame_dim})/2"

ffmpeg_args = ffmpeg_downscale_args+","+ffmpeg_crop_to_square_args

delete_files_in_directory(dst)
for root, dirs, filenames in os.walk(src, topdown=False):
    for filename in filenames:
        if ".flv" in filename:
            inputfile = os.path.join(root, filename)
            print(inputfile)
            outputfile = os.path.join(dst, filename.replace(".flv", ".mp4"))
            subprocess.run(['ffmpeg', '-i', inputfile,"-vf",ffmpeg_args,outputfile])
