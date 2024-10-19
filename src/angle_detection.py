import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# We are grabbing the pose estimation module from media pipe
mp_pose = mp.solutions.pose

# Setup mediapipe instance
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        print(mp_pose.POSE_CONNECTIONS)
        # Recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Making a detection
        results = pose.process(image)
        
        # Recoloring back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try: 
            landmarks = results.pose_landmarks.landmark

            def calculate_angle(a,b,c):
                a = np.array(a) # First 
                b = np.array(b) # Mid
                c = np.array(c) # End

                radians = np.arctan2(c[1]-b[1],c[0] - b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi) # converting radians to degrees

                if angle > 180.0:
                    angle = 360 - angle
                
                return angle

            # # Get coordinates for left shoulder, left elbow and left wrist
            # left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            # left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # # Calculate the angle
            # angle = calculate_angle(left_shoulder, left_elbow, left_wrist)


            # Get coordinates for left shoulder, left elbow and left wrist
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle2
            angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            
            # Get coordinates for left shoulder, left elbow and left wrist
            # right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            # right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            # Calculate the angle3
            angle3 = calculate_angle(right_elbow, right_shoulder, left_shoulder)


            # Get coordinates for left shoulder, left elbow and left wrist
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle2
            angle3 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # # Visualize angles
            # # Left arm
            # cv2.putText(image, str(int(angle)),
            #             tuple(np.multiply(left_elbow,[640,480]).astype(int)),
            #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA
            #                   )

            # Right arm- elbow
            cv2.putText(image, str(int(angle2)),
                        tuple(np.multiply(right_elbow,[640,480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA
                              )
            # Right arm- shoulder
            cv2.putText(image, str(int(angle3)),
                        tuple(np.multiply(right_shoulder,[640,480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2, cv2.LINE_AA
                              )
            

        except:
            pass

        # Render detections and setting the color for joints and edges
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
        
        cv2.imshow('Mediapipe feed', image)

        # waitKey collects the number of frames in ms and 0XFF searches for what key is pressed on the keyboard: Note 'q' != 'Q'
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            break 
        

    cap.release()
    cv2.destroyAllWindows()


    
