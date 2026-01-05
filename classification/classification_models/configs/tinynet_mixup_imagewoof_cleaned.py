# Configuration for training with cleaned dataset
# 533 problematic samples identified by Cleanlab have been removed
# Based on tinynet_mixup_imagewoof.py

# Import all settings from base config
_base_ = ['./tinynet_mixup_imagewoof.py']

# Change work directory to track cleaned version separately
work_dir = './work_dirs/tinynet_mixup_imagewoof_cleaned'

# Update WandB logging to differentiate cleaned training runs
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='imagewoof-160',
                name='tinynet_mixup_mmpretrain_cleaned',  # Different from original
            )
        )
    ]
)

# Note: The actual dataset filtering is handled by the training script (tinynet.py)
# using the clean_dataset.json file which contains:
# - total_samples: 12454
# - clean_samples: 11921
# - issue_samples: 533 (removed)

