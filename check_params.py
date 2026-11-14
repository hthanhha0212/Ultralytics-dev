import sys

from ultralytics import YOLO


def check_params(yaml_path):
    try:
        model = YOLO(yaml_path)
        # We can just get model.info()
        print(model.info())
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_params(sys.argv[1])
