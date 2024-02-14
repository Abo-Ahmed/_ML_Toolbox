import cv2
import os

def convert_video_to_images(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the first frame
    success, frame = video.read()
    count = 0

    while success:
        # Save the current frame as an image
        output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        # Read the next frame
        success, frame = video.read()
        count += 1

    # Release the video object
    video.release()

# Specify the path to the video file and the output folder
video_path = "J:\\task.mp4"
output_folder = "J:\\target"

# Call the function to convert the video to images
convert_video_to_images(video_path, output_folder)
