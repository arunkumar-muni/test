import cv2

class ROIVideoProcessor:
    def __init__(self):
        self.roi_selected = False
        self.roi = None
        self.selecting = False
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.output_video = None

    def select_roi(self, event, x, y, flags, param):
        """Mouse callback to handle ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)


        elif event == cv2.EVENT_MOUSEMOVE and self.selecting:
            self.end_point = (x, y)


        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.roi = (min(self.start_point[0], self.end_point[0]),
                        min(self.start_point[1], self.end_point[1]),
                        abs(self.start_point[0] - self.end_point[0]),
                        abs(self.start_point[1] - self.end_point[1]))
            self.roi_selected = True
            print(f"ROI selected: {self.roi}")

    def canny_edge_detection(self, frame):
        """Perform Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5)
        edges = cv2.Canny(blurred, 70, 135)
        return blurred, edges

    def process_video(self):
        """Main function to process the video stream."""
        cap = cv2.VideoCapture(0)

        cv2.namedWindow("Webcam Feed")
        cv2.setMouseCallback("Webcam Feed", self.select_roi)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Image not captured')
                break

            display_frame = frame.copy()

            if self.selecting:

                cv2.rectangle(display_frame, self.start_point, self.end_point, (0, 255, 0), 2)

            if self.roi_selected and self.roi:
                x, y, w, h = self.roi
                cropped_region = frame[y:y+h, x:x+w]
                blurred, edges = self.canny_edge_detection(cropped_region)

                if self.output_video is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.output_video = cv2.VideoWriter('edge_detected_output.avi', fourcc, 20.0, (w, h))

                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                self.output_video.write(edges_colored)

                cv2.imshow("Cropped ROI", cropped_region)
                cv2.imshow("Edges of ROI", edges)
            else:
                cv2.destroyWindow("Cropped ROI")
                cv2.destroyWindow("Edges of ROI")

            cv2.imshow("Webcam Feed", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting program.")
                break
            elif key == ord('r'):
                self.roi_selected = False
                self.roi = None
                print("ROI selection reset.")

        cap.release()
        if self.output_video is not None:
            self.output_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = ROIVideoProcessor()
    processor.process_video()






