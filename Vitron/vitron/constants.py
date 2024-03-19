CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ======================================================================================================
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"

# ======================================================================================================
OBJS_TOKEN_INDEX = -300
DEFAULT_OBJS_TOKEN = "<objs>"
DEFAULT_OBJS_PATCH_TOKEN = "<im_patch>"
DEFAULT_OBJS_START_TOKEN = "<objs_start>"
DEFAULT_OBJS_END_TOKEN = "<objs_end>"
OBJS_PLACEHOLDER = "<objs-placeholder>"
# ======================================================================================================

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620
