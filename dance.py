import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_similarity(pose_sample, pose_standard):
    angles_sample = []
    angles_standard = []

    # Calculate angles for right arm
    if pose_sample.pose_landmarks and pose_standard.pose_landmarks:
        angle_right_arm_sample = np.arctan2(pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                                            pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
        angle_right_arm_standard = np.arctan2(pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                                               pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
        angles_sample.append(angle_right_arm_sample)
        angles_standard.append(angle_right_arm_standard)

    # Calculate angles for left arm
    if pose_sample.pose_landmarks and pose_standard.pose_landmarks:
        angle_left_arm_sample = np.arctan2(pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                                           pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
        angle_left_arm_standard = np.arctan2(pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                                              pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
        angles_sample.append(angle_left_arm_sample)
        angles_standard.append(angle_left_arm_standard)

    # Calculate angles for right leg
    if pose_sample.pose_landmarks and pose_standard.pose_landmarks:
        angle_right_leg_sample = np.arctan2(pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                                            pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
        angle_right_leg_standard = np.arctan2(pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                                               pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
        angles_sample.append(angle_right_leg_sample)
        angles_standard.append(angle_right_leg_standard)

    # Calculate angles for left leg
    if pose_sample.pose_landmarks and pose_standard.pose_landmarks:
        angle_left_leg_sample = np.arctan2(pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                                           pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x - pose_sample.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
        angle_left_leg_standard = np.arctan2(pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                                              pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x - pose_standard.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
        angles_sample.append(angle_left_leg_sample)
        angles_standard.append(angle_left_leg_standard)

    if len(angles_sample) == 0 or len(angles_standard) == 0:
        return 0.0

    # Calculate similarity based on sum of squared differences (SSD)
    ssd = np.sum(np.square(np.array(angles_sample) - np.array(angles_standard)))
    similarity_score = 1 - (ssd / len(angles_sample))

    return similarity_score


def main():
    # Load sample and standard dance videos
    sample_video_path = 'Untitled video - Made with Clipchamp (1).mp4'
    standard_video_path = 'Untitled video - Made with Clipchamp (2).mp4'

    # Create video capture objects
    sample_video = cv2.VideoCapture(sample_video_path)
    standard_video = cv2.VideoCapture(standard_video_path)

    # Get screen resolution
    screen_width = int(sample_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(sample_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate half the screen size
    display_width = screen_width // 2
    display_height = screen_height // 2

    # Create pose detection models
    pose_detection = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize control variables
    paused = False
    current_frame = 0

    # Read frames from videos
    while True:
        if not paused:
            # Read frames from sample and standard videos
            ret_sample, frame_sample = sample_video.read()
            ret_standard, frame_standard = standard_video.read()

            if not ret_sample or not ret_standard:
                break

            # Increment the current frame count
            current_frame += 1

        # Resize frames to half the screen size
        frame_sample = cv2.resize(frame_sample, (display_width, display_height))
        frame_standard = cv2.resize(frame_standard, (display_width, display_height))

        # Convert frames to RGB
        frame_sample_rgb = cv2.cvtColor(frame_sample, cv2.COLOR_BGR2RGB)
        frame_standard_rgb = cv2.cvtColor(frame_standard, cv2.COLOR_BGR2RGB)

        # Detect poses in the frames
        results_sample = pose_detection.process(frame_sample_rgb)
        results_standard = pose_detection.process(frame_standard_rgb)

        # Draw pose landmarks on the frames
        annotated_sample = frame_sample.copy()
        mp_drawing.draw_landmarks(annotated_sample, results_sample.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        annotated_standard = frame_standard.copy()
        mp_drawing.draw_landmarks(annotated_standard, results_standard.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Calculate similarity between the sample and standard poses
        similarity = calculate_similarity(results_sample, results_standard)

        # Display the frames and similarity score
        similarity_score_percentage = map(similarity, -1, 1, 0, 100)
        similarity_score_percentage = max(0, min(100,
                                                 similarity_score_percentage))  # Clip the values to ensure it's within the range


        # cv2.imshow('Sample Pose', annotated_sample)
        cv2.imshow('Standard Pose', annotated_standard)
        if(similarity_score_percentage<90):
            cv2.putText(annotated_sample, f'Accuracy : {similarity_score_percentage:.2f}%         Score: {similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_sample,
                        f'Accuracy : {similarity_score_percentage:.2f}%      Score: {similarity:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sample pose', annotated_sample)

        # Check for key press events
        key = cv2.waitKey(1)
        if key == ord('p'):  # Pause or play the videos
            paused = not paused
        elif key == ord('r'):  # Restart the videos
            sample_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            standard_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
        elif key == ord('n'):  # Skip to the next frame
            sample_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            standard_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('f'):  # Forward by 2 seconds
            sample_video.set(cv2.CAP_PROP_POS_MSEC, sample_video.get(cv2.CAP_PROP_POS_MSEC) + 2000)
            standard_video.set(cv2.CAP_PROP_POS_MSEC, standard_video.get(cv2.CAP_PROP_POS_MSEC) + 2000)
            current_frame = int(sample_video.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == ord('b'):  # Reverse back by 2 seconds
            sample_video.set(cv2.CAP_PROP_POS_MSEC, sample_video.get(cv2.CAP_PROP_POS_MSEC) - 2000)
            standard_video.set(cv2.CAP_PROP_POS_MSEC, standard_video.get(cv2.CAP_PROP_POS_MSEC) - 2000)
            current_frame = int(sample_video.get(cv2.CAP_PROP_POS_FRAMES))

        # Exit loop if 'q' is pressed
        if key == ord('q'):
            break

        # Release video capture objects
    sample_video.release()
    standard_video.release()

    # Destroy OpenCV windows
    cv2.destroyAllWindows()


def map(value, start1, stop1, start2, stop2):
    """Maps a value from one range to another range."""
    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1))


# Run the main function
if __name__ == '__main__':
    main()
