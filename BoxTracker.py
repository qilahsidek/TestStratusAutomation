import cv2
import numpy as np
import matplotlib.pyplot as plt

#load video
video_path = 'BoxMovement.mp4'
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#segmentation model
model = cv2.dnn.readNetFromTensorflow('deeplabv3_xception65.pb')

#first frame
_, frame = cap.read()
blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
model.setInput(blob)
output = model.forward()
segmentation_mask = np.argmax(output.squeeze(), axis=0)

#box detection
box_class_index = 1
object_mask = np.uint8(segmentation_mask == box_class_index)  # Replace class_id with the appropriate class index for the box
contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])  # Assuming only one contour exists

#tracker
tracker = cv2.TrackerCSRT_create()  # You can choose a different tracker if desired
bbox = (x, y, w, h)
tracker.init(frame, bbox)

#track the box
tracked_frames = []
speed_data = []

while True:
    success, frame = cap.read()
    if not success:
        break
    
    _, segmented_frame = cap.read()  # Separate frame for segmentation visualization
    
    # Track the box
    success, bbox = tracker.update(frame)
    
    if success:
        # Extract the tracked box coordinates
        x, y, w, h = map(int, bbox)
        
        # Draw the tracked rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the speed (distance moved) of the box in pixels per frame
        speed = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        speed_data.append(speed)
        
        # Update previous position
        prev_x, prev_y = x, y
        
        # Store segmented frame at specific intervals
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == 20 or cap.get(cv2.CAP_PROP_POS_FRAMES) == 60:
            tracked_frames.append(segmented_frame)
    else:
        break

#outputs
# 1) Tracked Video
output_video = cv2.VideoWriter('tracked_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height), isColor=True)
for frame in tracked_frames:
    output_video.write(frame)
output_video.release()

# 2) Segmented Box Images
output_images = ['first_frame_segmented_box.png', 'frame_20_segmented_box.png', 'frame_60_segmented_box.png']
for i, frame in enumerate(tracked_frames):
    cv2.imwrite(output_images[i], frame)

# 3) Speed vs. Time Graph
frame_indices = np.arange(len(speed_data))
plt.plot(frame_indices, speed_data)
plt.xlabel('Frame')
plt.ylabel('Speed (pixels per frame)')
plt.title('Box Speed vs. Time')
plt.savefig('speed_vs_time.png')
plt.show()
