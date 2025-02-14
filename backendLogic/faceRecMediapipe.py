import cv2
import time
import os
import csv
import mediapipe as mp


class FaceAnalyzer:
    def __init__(self, video_path,
                 output_csv="face_data.csv",
                 enable_detection=True,
                 enable_mesh=True,
                 max_faces=1):
        self.video_path = video_path
        self.output_csv = output_csv
        self.cap = cv2.VideoCapture(video_path)
        self._validate_video()

        self.csv_file = open(output_csv, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self._init_csv_header()

        self.enable_detection = enable_detection
        self.enable_mesh = enable_mesh

        # Init MediaPipe
        self.mp_draw = mp.solutions.drawing_utils
        self._init_models(max_faces)

        # trace
        self.frame_count = 0
        self.ptime = 0

    def _validate_video(self):
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Not able to open: {self.video_path}")

    def _init_csv_header(self):
        """Init csv header"""
        headers = ['frame', 'timestamp', 'face_id', 'landmark_id', 'x', 'y']
        self.csv_writer.writerow(headers)

    def _init_models(self, max_faces):
        """Init MediaPipe"""
        if self.enable_detection:
            self.detector = mp.solutions.face_detection.FaceDetection()

        if self.enable_mesh:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mesh_spec = mp.solutions.drawing_utils.DrawingSpec(
                color=(180, 180, 180),
                thickness=1,
                circle_radius=1
            )

    def _process_mesh(self, frame):
        """process mesh"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_id, landmarks in enumerate(results.multi_face_landmarks):
                self._record_landmarks(frame, face_id, landmarks)
                self._draw_mesh(frame, landmarks)
        return frame

    def _record_landmarks(self, frame, face_id, landmarks):
        """记录特征点到CSV"""
        h, w = frame.shape[:2]
        timestamp = time.time()

        for lm_id, landmark in enumerate(landmarks.landmark):
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)

            # write to csv
            self.csv_writer.writerow([
                self.frame_count,
                timestamp,
                face_id,
                lm_id,
                cx,
                cy
            ])

            # output
            print(f"Frame {self.frame_count} | Landmark {lm_id}: ({cx}, {cy})")

    def _draw_mesh(self, frame, landmarks):
        """绘制面部网格"""
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mesh_spec
        )

    def _update_fps(self, frame):
        """更新并显示FPS"""
        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if (ctime - self.ptime) > 0 else 0
        self.ptime = ctime
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def run_analysis(self):
        """主分析循环"""
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 处理面部网格
                if self.enable_mesh:
                    frame = self._process_mesh(frame)

                # update frame
                self._update_fps(frame)

                # show results
                cv2.imshow('Face Analysis', frame)
                self.frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self._cleanup()

    def _cleanup(self):
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
        print(f"分析完成！数据已保存至: {self.output_csv}")


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "videoClip", "02-01-04-02-02-02-04.mov")
    output_csv = os.path.join(current_dir, "analysis_results.csv")


    analyzer = FaceAnalyzer(
        video_path=video_path,
        output_csv=output_csv,
        enable_detection=False,
        enable_mesh=True,
        max_faces=1
    )


    try:
        analyzer.run_analysis()
    except Exception as e:
        print(f"发生错误: {str(e)}")

        #   Filename identifiers
        #
        # Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        # Vocal channel (01 = speech, 02 = song).
        # Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        # Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
        # Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
        # Repetition (01 = 1st repetition, 02 = 2nd repetition).
        # Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
        # Filename example: 02-01-06-01-02-01-12.mp4
        #
        # Video-only (02)
        # Speech (01)
        # Fearful (06)
        # Normal intensity (01)
        # Statement “dogs” (02)
        # 1st Repetition (01)
        # 12th Actor (12)
        # Female, as the actor ID number is even.