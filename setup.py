from setuptools import setup, find_packages
setup(
    name="mnn", version="0.8.0",
    description="Mathematical Neural Network Framework — Research Grade",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)
