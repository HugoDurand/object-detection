from imageai.Detection import ObjectDetection

detector = ObjectDetection()

input_path = './input/input.jpg'
model_path = './models/yolo-tiny.h5'
# model_path = './models/resnet50_coco_best_v2.0.1.h5'
# model_path = './models/yolo.h5'
output_path = './output/output.jpg'

detector.setModelTypeAsTinyYOLOv3()
# detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsRetinaNet()

detector.setModelPath(model_path)

detector.loadModel()

detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for item in detection:
    print(item["name"], " : ", item["percentage_probability"])
