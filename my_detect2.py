import os
import glob
import skimage.io

from config import Configs
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "myshape"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

inference_config.display()

def create_file_lists_from_path(file_dir, pattern='*.tif'):
    """
    获取file_dir下pattern的路径和文件名。

    目录结构为file_dir/*.tif

    :param file_dir: 保存文件路径
    :param postfix: 文件的匹配格式
    :return: 一个列表，列表中是文件记录的字典，{'file': 文件路径, 'filename': 文件名}
    """
    if not os.path.exists(file_dir):
        print("File directory '" + file_dir + "' not found.")
        return None
    files_list = []

    file_list = [] # 保存路径下文件的路径
    file_glob = os.path.join(file_dir, pattern)
    file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        for f in file_list:
            filename = os.path.splitext(f.split("/")[-1])[0]
            record = {'file': f, 'filename': filename}
            files_list.append(record)
    print('Found ' + str(len(file_list)) + ' pictures in path ' + os.path.join(os.getcwd(), file_dir))

    return files_list

# Prepare the test data
TEST_images_DIR = os.path.join(ROOT_DIR, inference_config.NAME+"_test", "images")
TEST_results_DIR = os.path.join(ROOT_DIR, inference_config.NAME+"_test", "results")

files_list = create_file_lists_from_path(TEST_images_DIR, '*.tif')

images = []
for i in N:
    image = skimage.io.imread(files_list['file'])
    images.append(image)

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

results = model.detect(images, verbose=1)
results['image_name'] = files_list['filename']

with open(os.path.join(TEST_results_DIR, 'results.pickle'), 'wb') as file:
    pickle.dump(results, file)