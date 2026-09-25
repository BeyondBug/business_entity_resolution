import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_S1 = os.path.join(BASE_DIR, "../student_resource/dataset/train/train_source1.tsv")
TRAIN_S2 = os.path.join(BASE_DIR, "../student_resource/dataset/train/train_source2.tsv")
TRAIN_S3 = os.path.join(BASE_DIR, "../student_resource/dataset/train/train_source3.tsv")
TRAIN_GT = os.path.join(BASE_DIR, "../student_resource/dataset/train/train_ground_truth.tsv")

TEST_S1  = os.path.join(BASE_DIR, "../student_resource/dataset/test/test_source1.tsv")
TEST_S2  = os.path.join(BASE_DIR, "../student_resource/dataset/test/test_source2.tsv")
TEST_S3  = os.path.join(BASE_DIR, "../student_resource/dataset/test/test_source3.tsv")

MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EXP_DIR    = os.path.join(BASE_DIR, "experiments")

MIN_TOKEN_LEN       = 3
MIN_TOKEN_OVERLAP   = 2
MIN_BIGRAM_OVERLAP  = 3
MIN_ADDR_TOKEN_LEN  = 4
MIN_NUM_LEN         = 4
TFIDF_TOP_K_NAME    = 15
TFIDF_TOP_K_ADDR    = 8
TFIDF_MIN_SIM_NAME  = 0.25
TFIDF_MIN_SIM_ADDR  = 0.20

RANDOM_SEED         = 42
VAL_SIZE            = 0.20
XGB_N_ESTIMATORS    = 800
XGB_MAX_DEPTH       = 6
XGB_LR              = 0.03
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE       = 0.8
HARD_NEG_THRESHOLD  = 0.65

SINGLETON_THRESHOLD = 0.35
ACCEPT_THRESHOLD    = 0.55
