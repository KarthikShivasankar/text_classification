from setuptools import setup, find_packages

setup(
    name="text_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "transformers>=4.5.0",
        "datasets>=1.5.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.1.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "GPUtil>=1.4.0",
        "psutil>=5.8.0",
        "wandb>=0.12.0",
        "tqdm>=4.62.0",
        "codecarbon>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-binary=text_classifier.train_binary:main",
            "train-multiclass=text_classifier.train_multiclass:main",
            "train-binary-multiple=text_classifier.train_binary_multiple:main",
        ],
    },
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for fine-tuning transformer models for text classification tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text_classifier",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
