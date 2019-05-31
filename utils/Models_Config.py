HIGH_RESOLUTION_NET = [
    {'PRETRAINED_LAYERS': ['*'],
     'STEM_INPLANES':64,
     'FINAL_CONV_KERNEL':1,
     'WITH_HEAD':True},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 2,
     'NUM_BLOCKS': [4, 4],
     'NUM_CHANNELS': [32, 64],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM'},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 3,
     'NUM_BLOCKS': [4, 4, 4],
     'NUM_CHANNELS': [32, 64, 128],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM'},

    {'NUM_MODULES': 1,
     'NUM_BRANCHES': 4,
     'NUM_BLOCKS': [4, 4, 4, 4],
     'NUM_CHANNELS': [32, 64, 128, 256],
     'BLOCK': 'BASIC',
     'FUSE_METHOD': 'SUM'},
]

