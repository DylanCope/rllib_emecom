import setuptools
import codecs
import os


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setuptools.setup(
    name='rllib_emecom',
    version='{{VERSION_PLACEHOLDER}}',
    author="Dylan R. Cope",
    description='Framework for Emergent Communication using RLlib',
    url='https://github.com/DylanCope/rllib_emecom',
    long_description_content_type='text/markdown',
    long_description=long_description,
    packages=setuptools.find_packages(where='.'),
    install_requires=[
        'ray[rllib]>=2.7.0rc0',
        'torch>=2.1.0',
        'gymnasium>=0.28.1',
        'pettingzoo>=1.24.1',
        'numpy>=1.23.5',
        'pandas>=1.3.5',
        'networkx>=2.8.8',
        'matplotlib>=3.6.2',
        'imageio[ffmpeg]>=2.31.3',
        'seaborn>=0.11.2',
    ]
)
