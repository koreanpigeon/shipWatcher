import torch
import cv2
from PIL import Image
from model_training import pic_transform
from model_training import get_shipWatcher
from model_training import class_list


device = torch.device("mps")
shipWatcher = get_shipWatcher()
state_dict = torch.load("shipWatcher.pth", map_location=device)
shipWatcher.load_state_dict(state_dict)
shipWatcher.to(device)
shipWatcher.eval()


@torch.no_grad()
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image = Image.fromarray(rgb_frame)
    rgb_tensor = pic_transform(rgb_image).unsqueeze(0)
    rgb_tensor = rgb_tensor.to(device)
    class_prediction = torch.argmax(shipWatcher(rgb_tensor), dim=1).item()
    if class_prediction != 0:
        print(f"Warning! {class_list[class_prediction]} detected!")
        image = Image.fromarray(rgb_frame)
        image.show()
    else:
        print("No threats detected.")


def process_video(video_path, skip_frames = 30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    else:
        print("Processing video... Press 'q' to stop.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing finished")
            break
        if frame_count % skip_frames == 0:
            process_frame(frame)
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()




