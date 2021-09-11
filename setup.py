from distutils.core import setup

setup(
    name='ephys_analysis',
    version='0.1.0',
    author='Nolan lab',
    package=['ephys_analysis'],
    install_requires=[
        'matplotlib',
        'pandas',
        'scipy',
        'pyyaml',
        'pytest',
       ' ml_ms4alg',
        'pandas',
        'joblib',
        'scipy',
        'matplotlib',
        'numba',
        'xlrd',
        'scikit-image',
        'cmocean',
        'pyyaml',
        'tqdm',
        'spikeforest',
        'palettable',
        'imbalanced-learn'
    ],
    scripts=['runSnake']
)