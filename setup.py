"""
Setup script for Respiratory Sound Analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="respiratory-sound-analysis",
    version="1.0.0",
    author="Rajasekar SS, Manal Othman, Karthiga M, et al.",
    author_email="your.email@example.com",
    description="Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/respiratory-sound-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "respiratory-analysis=src.main:main",
        ],
    },
)
