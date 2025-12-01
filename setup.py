from setuptools import setup, find_packages

setup(
    name="sparse_dueling_bandits",
    version="1.0.0",
    description="Sparse Feature Dueling Bandits for Copeland Winner Identification",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "sparse-bandits=run_experiment:main",
        ],
    },
)
