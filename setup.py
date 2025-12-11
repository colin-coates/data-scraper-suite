"""
Data Scraper Suite for Mountain Jewels Intelligence

A comprehensive web scraping and data collection platform with anti-detection
capabilities, plugin architecture, and enterprise-grade reliability.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mj-data-scraper-suite",
    version="1.0.0",
    author="Mountain Jewels Intelligence",
    author_email="engineering@mountainjewels.com",
    description="Enterprise web scraping platform with anti-detection and plugin architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mountain-jewels-intelligence/data-scraper-suite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["black", "isort", "flake8", "mypy"],
        "test": ["pytest-cov", "pytest-xdist"],
    },
    entry_points={
        "console_scripts": [
            "mj-scraper=mj_data_scraper_suite.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
