## feature parameters
# 1s = 43frame
SR = 22050
N_FFT = 2048
N_HOP = 512
N_MEL = 128
H_MEL = 81

## data devide
CHUNK_SIZE = 43
N_CHUNK = 72
N_FRAME = N_CHUNK * CHUNK_SIZE
FRAME_PER_SEC = SR / N_HOP
SEC_PER_CHUNK = round(CHUNK_SIZE / FRAME_PER_SEC)

## training parameters
SHAPE = (N_MEL, None, 1)
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 1000
PATIENCE = 100
VALIDATION_STEPS = 20

## data path
FEATURE_PATH = '../Datasets/feature/'
ANNOTATION_PATH = '../Datasets/chorus_label/'


## training files
feature_files = [
    # FEATURE_PATH + 'Harmonix.joblib',
    # FEATURE_PATH + 'Beatles.joblib',
    # FEATURE_PATH + 'MJ.joblib',
]

train_annotation_file = [
    # ANNOTATION_PATH + 'Beatles.joblib',
    # ANNOTATION_PATH + 'MJ.joblib',
    # ANNOTATION_PATH + 'Harmonix.joblib',
]


## valid files
valid_feature = [
    # FEATURE_PATH + 'Validation.joblib',
]
valid_annotation_file = [
    # ANNOTATION_PATH + 'Validation.joblib',
]


## testing files
test_annotation_files = [
    # ANNOTATION_PATH + 'RWC.joblib',
    # ANNOTATION_PATH + 'SP.joblib',
    # ANNOTATION_PATH + 'SL.joblib',
]
test_feature_files = [
    # FEATURE_PATH + 'RWC.joblib',
    # FEATURE_PATH + 'SP.joblib',
    # FEATURE_PATH + 'SL.joblib',
]
