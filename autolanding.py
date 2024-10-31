import time
import cv2
import numpy as np
from djitellopy import Tello


class AutoLandingTello:
    def __init__(self):
        # Initialize the Tello drone object
        self.tello = Tello()
        # self.camera_matrix = np.load('./mtx.npy')
        # self.distortion = np.load('./dist.npy')
        self.camera_matrix = np.array(
            [
                [921.170702, 0.000000, 459.904354],
                [0.000000, 919.018377, 351.238301],
                [0.000000, 0.000000, 1.000000],
            ]
        )
        self.distortion = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.14  # ArUco marker size in meters (14cm)
        self.last_tvec = None  # Attribute to store the last detected ArUco marker position
        self.centralize_threshold = 0.1  # 10cm threshold for centralizing (x,y axes)
        self.distance_threshold = 0.30  # 30cm threshold for forward distance (z axis)

    def get_position(self):
        # Get a frame capture from the drone's camera
        frame = self.tello.get_frame_read().frame

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the frame
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            # Estimate pose of the marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix,
                                                                  self.distortion)
            # Store and return the first detected marker's tvec
            self.last_tvec = tvecs[0][0]
            return self.last_tvec
        else:
            if self.last_tvec is not None and abs(self.last_tvec[2]) < 30:
                # If no marker is detected, and we were very close to it
                # return the last known position
                return self.last_tvec

            # assuming ArUco code was moved
            return None

    def draw_axis(self, frame, tvec):
        try:
            # Axis length (in meters)
            axis_length = 0.1

            # Define the axis points
            axis_points = np.float32(
                [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)

            # Project axis points
            rvecs, _ = cv2.Rodrigues(np.zeros(3))
            axis_points, _ = cv2.projectPoints(axis_points, rvecs, tvec, self.camera_matrix, self.distortion)

            # Convert to tuple of integers
            axis_points = axis_points.reshape(-1, 2).astype(int)

            # Draw axis lines
            frame = cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[1]), (0, 0, 255), 2)  # X axis (red)
            frame = cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[2]), (0, 255, 0), 2)  # Y axis (green)
            frame = cv2.line(frame, tuple(axis_points[0]), tuple(axis_points[3]), (255, 0, 0), 2)  # Z axis (blue)

            # Add text overlay with tvec information
            text = f"ArUco marker Position x={tvec[0]:.2f} y={tvec[1]:.2f} z={tvec[2]:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception:
            pass

        return frame

    def run(self):
        # Connect to the Tello drone
        self.tello.connect()

        # Start video stream
        self.tello.streamon()

        # Print battery status to confirm connection
        battery_level = self.tello.get_battery()
        print(f"Battery level: {battery_level}%")

        # Take off
        self.tello.takeoff()

        # Safe sleep of 2 seconds to ensure stability
        time.sleep(2)

        print("Drone has taken off, video stream is on, and it's ready for further operations.")

    def centralize(self, position):
        if position is None:
            return False

        x_distance = position[0]

        if abs(x_distance) < self.centralize_threshold:
            return True

        # Adjust drone's position left or right based on x distance
        move_distance = 20  # Move distance in cm

        if x_distance > 0:
            self.tello.move_right(move_distance)
            self.last_tvec[0] -= move_distance / 100  # Adjust last_tvec in meters
        else:
            self.tello.move_left(move_distance)
            self.last_tvec[0] += move_distance / 100  # Adjust last_tvec in meters

        return False

    def move_towards_target(self, position):
        if position is None:
            return False

        x_distance = position[0]
        y_distance = position[1]
        z_distance = position[2]

        if (abs(x_distance) <= self.centralize_threshold and abs(y_distance) <= self.centralize_threshold and
                abs(z_distance) <= self.distance_threshold):
            # x,y,z are close to target, we're on the spot!
            return True

        move_distance = 20  # Move distance in cm

        # Move forward if z distance is greater than threshold
        if abs(z_distance) > self.distance_threshold:
            self.tello.move_forward(move_distance)
            self.last_tvec[2] -= move_distance / 100  # Adjust last_tvec in meters

        # Move down if y distance is greater than threshold
        if abs(y_distance) > self.centralize_threshold:
            self.tello.move_down(move_distance)
            self.last_tvec[1] -= move_distance / 100  # Adjust last_tvec in meters

        return False

    def scan(self):
        max_height = 100  # Maximum height in cm
        min_height = 40  # Minimum height in cm

        # Move the drone down to the minimum height
        while True:
            current_height = self.tello.get_height()
            if current_height <= min_height:
                break
            self.tello.move_down(20)
            time.sleep(1)

        height = self.tello.get_height()
        while height < max_height:
            for _ in range(12):  # 360 degree rotation in 30-degree increments
                self.tello.rotate_clockwise(30)
                time.sleep(1)
                position = self.get_position()
                if position is not None:
                    print(f"ArUco marker found at x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
                    return position

            self.tello.move_up(20)
            height = self.tello.get_height()
            time.sleep(1)

        print("No ArUco marker detected within 1 meter.")
        return None

    def auto_land(self):
        while True:
            # Get a frame capture from the drone's camera
            frame = self.tello.get_frame_read().frame

            # Display the frame
            cv2.imshow('Tello Video Feed', frame)

            # Get position of ArUco marker
            position = self.get_position()
            if position is not None:
                print(f"ArUco marker position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")

                # Draw axis on the frame
                frame = self.draw_axis(frame, position)

                # Display updated frame with axis
                cv2.imshow('Tello Video Feed', frame)

                # Centralize the drone
                if self.centralize(position):
                    print("Drone centered. Moving forward the target...")
                    if self.move_towards_target(position):
                        print("~~ ╔═════════════════════════════════════════════════════════╗ ~~")
                        print("~~ ║ Detected precise position. AUTO LANDING SUCCEEDED :)... ║ ~~")
                        print("~~ ╚═════════════════════════════════════════════════════════╝ ~~")
                        self.tello.land()  # Initiate landing
                        break
            else:
                print("No ArUco marker detected.")
                self.scan()  # Start scanning if no marker is detected

            # Check for key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Delay to control frame rate
            time.sleep(0.05)


        # Add a delay to ensure landing completes (adjust as needed)
        time.sleep(5)
        self.tello.streamoff()
        self.tello.end()

        # Close OpenCV windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    drone = AutoLandingTello()
    drone.run()
    drone.auto_land()
