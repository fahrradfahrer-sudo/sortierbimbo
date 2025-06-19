import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create an InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200  # Use HD1200 resolution
    init_params.camera_fps = 30  # Set FPS to 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use NEURAL depth mode
    init_params.coordinate_units = sl.UNIT.METER # Use meters for depth units

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error {err}, shutting down.")
        # Consider raising an exception or using a more robust error handling
        if zed.is_opened():
            zed.close()
        exit(1)

    print("ZED Camera opened successfully")

    # Create sl.Mat objects to store image and depth map
    image_sl = sl.Mat()
    depth_map_sl = sl.Mat()
    # Depth confidence can be managed using runtime_parameters.confidence_threshold
    # and runtime_parameters.texture_confidence_threshold.
    # The confidence map itself can be retrieved using sl.MEASURE.CONFIDENCE.
    runtime_parameters = sl.RuntimeParameters()


    while True:
        # Grab an image
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image_sl, sl.VIEW.LEFT)
            # Retrieve depth map
            zed.retrieve_measure(depth_map_sl, sl.MEASURE.DEPTH)

            # Convert image to OpenCV format
            image_ocv = image_sl.get_data()

            # Convert depth map to OpenCV format
            depth_map_ocv = depth_map_sl.get_data()

            # Display the image
            if image_ocv.shape[0] > 0 and image_ocv.shape[1] > 0:
                cv2.imshow("ZED Image", image_ocv)
            else:
                print("Retrieved empty image")

            # Process and display the depth map
            if depth_map_ocv.shape[0] > 0 and depth_map_ocv.shape[1] > 0:
                # Normalize the depth map for display
                # Replace NaN and Inf with 0 (or a large value if appropriate for your normalization)
                depth_map_ocv_display = np.nan_to_num(depth_map_ocv, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize to 0-255 range for 8-bit display
                # This normalization can be tricky. If max_val is 0 or very small, this might not work well.
                # A more robust normalization might consider a fixed max depth or percentile.
                min_val, max_val = np.min(depth_map_ocv_display), np.max(depth_map_ocv_display)
                if max_val > min_val: # Avoid division by zero if depth is all one value
                    depth_map_normalized = ((depth_map_ocv_display - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                else: # If all depth values are the same (e.g., all 0s or all Infs that became 0)
                    depth_map_normalized = np.zeros(depth_map_ocv_display.shape, dtype=np.uint8)

                cv2.imshow("ZED Depth", depth_map_normalized)
            else:
                print("Retrieved empty depth map")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # print("Failed to grab image/depth") # Less verbose
            key = cv2.waitKey(1) & 0xFF # Still allow exit if grab fails
            if key == ord('q'):
                break
            # import time
            # time.sleep(0.01)


    # Close the camera
    zed.close()
    cv2.destroyAllWindows()
    print("ZED Camera closed.")

if __name__ == "__main__":
    main()
