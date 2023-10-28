import setuptools


setuptools.setup(
    name="rllib_emecom",
    version="0.0.1",
    author="Dylan R. Cope",
    description="Framework for Emergent Communication using RLlib",
    packages=["rllib_emecom"],
    install_requires=[
        'ray[rllib]>=2.7.0rc0',
        'torch>=2.1.0',
        'gymnasium>=0.28.1',
        'pettingzoo>=1.24.1'
        'numpy>=1.23.5',
        'pandas>=1.3.5',
        'networkx>=2.8.8',
        'matplotlib>=3.6.2',
        'imageio[ffmpeg]>=2.31.3',
        'seaborn>=0.11.2',
    ]
)
