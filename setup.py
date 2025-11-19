"""
Setup script for SiC Power Module Thermal Lifetime Prediction Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sic-thermal-lifetime",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AQG 324 compliant thermal lifetime prediction for SiC power modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sic-thermal-lifetime",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/sic-thermal-lifetime/issues",
        "Documentation": "https://github.com/yourusername/sic-thermal-lifetime#readme",
        "Source Code": "https://github.com/yourusername/sic-thermal-lifetime",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="power electronics, thermal fatigue, lifetime prediction, AQG 324, SiC MOSFET, reliability",
    py_modules=["thermal_lifetime_prediction"],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "matplotlib>=3.2.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thermal-lifetime=thermal_lifetime_prediction:run_lifetime_test",
        ],
    },
)