"""
Setup script for Memorizer Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read dev requirements
with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read optional requirements
with open("requirements-optional.txt", "r", encoding="utf-8") as fh:
    optional_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="memorizer",
    version="1.0.0",
    author="Memorizer Team",
    author_email="team@memorizer.ai",
    description="Production-ready memory lifecycle framework for AI assistants and agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/memorizer-ai/memorizer",
    project_urls={
        "Bug Reports": "https://github.com/memorizer-ai/memorizer/issues",
        "Source": "https://github.com/memorizer-ai/memorizer",
        "Documentation": "https://docs.memorizer.ai",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "optional": optional_requirements,
        "all": dev_requirements + optional_requirements,
    },
    entry_points={
        "console_scripts": [
            "memorizer=memorizer.cli.memorizer_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "memorizer": [
            "*.yaml",
            "*.yml",
            "*.json",
            "templates/*",
            "configs/*",
        ],
    },
    keywords=[
        "ai",
        "memory",
        "framework",
        "agents",
        "llm",
        "langchain",
        "llamaindex",
        "autogpt",
        "crewai",
        "vector-database",
        "embeddings",
        "retrieval",
        "lifecycle",
    ],
)