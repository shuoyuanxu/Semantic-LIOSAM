"""Configuration for the forestry RandLA-Net model.

Values must match the ones used at training time
(originally class ConfigeveryCar_liu in the research code).
"""


class Config:
    name = 'Cunjia_liu'
    k_n = 10                          # KNN neighbours
    num_layers = 4                    # encoder layers
    num_points = 20480                # points per inference patch
    num_classes = 4
    sub_grid_size = 0.04              # grid subsampling size used at training
    batch_size = 4
    val_batch_size = 1
    train_steps = 300
    val_steps = 5
    sub_sampling_ratio = [4, 4, 4, 4]
    d_out = [16, 64, 128, 256]
    d_input_dim = [960, 448, 192, 64]
    d_input_shen_dim = [2496, 1216, 576, 224]
    d_output_dim = [1024, 512, 256, 128]

    noise_init = 3.5
    max_epoch = 50
    learning_rate = 1e-2
    lr_decays = {i: 0.95 for i in range(0, 500)}
    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    # filled in at runtime (kept for RandLANet.py compatibility)
    ignored_label_inds = []
    class_weights = None


# semantic classes of the trained checkpoint
LABEL_TO_NAME = {
    0: 'trunk',
    1: 'crown',
    2: 'ground',
    3: 'others',
}

# RGB (0-255) used when exporting coloured maps
LABEL_TO_COLOR = {
    0: (139, 69, 19),    # trunk  - brown
    1: (34, 139, 34),    # crown  - green
    2: (128, 128, 96),   # ground - olive grey
    3: (170, 170, 170),  # others - light grey
}

# near-field crop the network was trained on (sensor frame, metres)
TRAIN_CROP = {
    'x_min': -2.0, 'x_max': 3.0,
    'y_min': -4.0, 'y_max': 4.0,
}
