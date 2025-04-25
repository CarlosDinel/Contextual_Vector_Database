from setuptools import setup, find_packages

setup(
    name="contextual_vector_database",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sentence-transformers",
        "torch",
        "torchvision",
        "pillow",
        "scikit-learn"
    ],
    python_requires=">=3.8",
)
