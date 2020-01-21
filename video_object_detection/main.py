from imageai.Detection import VideoObjectDetection

detector = VideoObjectDetection()

input_path = './input/input.mp4'
# model_path = '../models/yolo-tiny.h5'
# model_path = '../models/resnet50_coco_best_v2.0.1.h5'
model_path = '../models/yolo.h5'
output_path = './output/output.mp4'

# detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsRetinaNet()

detector.setModelPath(model_path)

detector.loadModel()

detection = detector.detectObjectsFromVideo(input_file_path=input_path,
                                            output_file_path=output_path,
                                            frames_per_second=29,
                                            log_progress=True)
