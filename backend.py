from flask import Flask, render_template, Response, request
from cvzone.PoseModule import PoseDetector
import cv2

app = Flask(__name__)

# Initialize the PoseDetector object
detector = PoseDetector()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Flag to indicate if the user is in frame
user_in_frame = True

# Load the default t-shirt image
tshirt_img = cv2.imread("C:/Users/sriva/hci/E-Commerce-website-design-main/images/1.png",cv2.IMREAD_UNCHANGED)  # Replace "x.png" with the path to your t-shirt image

def generate_frames():
    global user_in_frame
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for intuitive interaction
        frame = cv2.flip(frame, 1)

        # Detect the pose in the frame
        frame = detector.findPose(frame)
        lmList, _ = detector.findPosition(frame, draw=False)

        if lmList:  # Check if pose landmarks are detected
            # Extract shoulder points (landmarks 11 and 12 for the left shoulder, 23 and 24 for the right shoulder)
            left_shoulder = lmList[11] if lmList[11] else (0, 0)
            right_shoulder = lmList[12] if lmList[12] else (0, 0)

            # Overlay t-shirt image onto the frame
            if all(left_shoulder) and all(right_shoulder):
                # Calculate the width and height of the t-shirt based on the shoulder coordinates
                tshirt_width = int(abs(right_shoulder[0] - left_shoulder[0]))
                tshirt_height = int(tshirt_width * (tshirt_img.shape[0] / tshirt_img.shape[1]))

                # Resize the t-shirt image to match the calculated width and height
                resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

                top_left = (int(min(left_shoulder[0], right_shoulder[0])), int(min(left_shoulder[1], right_shoulder[1])))

                # Ensure the top left coordinates are within the frame
                top_left = (max(top_left[0], 0), max(top_left[1], 0))

                # Calculate the bottom right coordinates
                bottom_right = (top_left[0] + tshirt_width, top_left[1] + tshirt_height)

                # Ensure the bottom right coordinates are within the frame
                bottom_right = (min(bottom_right[0], frame.shape[1]), min(bottom_right[1], frame.shape[0]))

                # Check if the ROI dimensions are greater than zero
                if bottom_right[0] - top_left[0] > 0 and bottom_right[1] - top_left[1] > 0:
                    # Resize the t-shirt image to match the ROI dimensions
                    resized_tshirt = cv2.resize(tshirt_img, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))

                    # Overlay the resized t-shirt image onto the frame
                    for c in range(0, 3):
                        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c] = resized_tshirt[:,:,c] * (resized_tshirt[:,:,3]/255.0) +  frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], c] * (1.0 - resized_tshirt[:,:,3]/255.0)


                # Check if user is out of frame
                if left_shoulder[0] == 0 or right_shoulder[0] == 0:
                    user_in_frame = False
                else:
                    user_in_frame = True

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    global user_in_frame
    # Get the image path from the query parameters
    image_src = request.args.get('image')

    # Read the image
    if image_src:
        image = cv2.imread(image_src)
        if image is None:
            print(f"Failed to read image at path: {image_src}")
        else:
            print("Image loaded successfully.")
    else:
        print("No image path provided.")

    return render_template('index.html', user_in_frame=user_in_frame,image_src = image_src)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
