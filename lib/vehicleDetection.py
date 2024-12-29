from ultralytics import YOLO
import os
import cv2
import argparse

def detect_vehicles(image_path, conf_threshold=0.3):
    # Load YOLOv8 model
    model = YOLO('yolo11n.pt')

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    # List of vehicle classes in COCO dataset
    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

    # Run detection
    results = model(image)

    # Counter for vehicles
    vehicle_count = 0

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = result.names[int(box.cls[0])]

            # Check if detected object is a vehicle
            if cls in vehicle_classes:
                # Get confidence score
                conf = float(box.conf[0])

                # Continue only if confidence is above threshold
                if conf >= conf_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label with class name and confidence
                    label = f"{cls} {conf * 100:.1f}%"
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    vehicle_count += 1

    # Add total vehicle count
    cv2.putText(image, f"Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, vehicle_count


def main(image_path, conf_threshold=0.3, dest_path=None):

    if conf_threshold < 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")

    if not image_path:
        raise ValueError("Image path is required")

    try:
        # Process image
        result_image, vehicle_count = detect_vehicles(image_path, conf_threshold)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Save result
        file_extension = image_path.split(".")[-1]
        image_name = image_path.split("/")[-1].split(".")[0]
        output_path = dest_path
        if not dest_path:
            output_path = f"{image_path.split('.')[0]}_result.{file_extension}"
        else:
            output_path = f"{output_path}{image_name}_result.{file_extension}"
        cv2.imwrite(output_path, result_image)
        print(f"Detection completed. Result saved to {output_path}")

        # Display result (optional)
        cv2.imshow("Vehicle Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Function to count vehicles in an image (e.g. for API)
def count_vehicle(image_path, conf_threshold=0.3):
    if conf_threshold < 0 or conf_threshold > 1:
        raise ValueError("Confidence threshold must be between 0 and 1")

    if not image_path:
        raise ValueError("Image path is required")

    try:
        # Process image
        result_image, vehicle_count = detect_vehicles(image_path, conf_threshold)

        return vehicle_count

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect vehicles in an image")
    parser.add_argument("--image", required=True, help="Path to image (include file extension)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold between 0 and 1")
    parser.add_argument("--dest", type=str, default=None, help="Path to save result image")

    args = parser.parse_args()

    if not args.dest.endswith("/"):
        args.dest += "/"

    # Run main function
    main(args.image, args.conf, args.dest)