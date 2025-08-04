"""
Setup script for Kallos Portfolios - Cryptocurrency Portfolio Construction & Backtesting Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "pandas>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyPortfolioOpt>=1.5.4",
        "cvxpy>=1.4.0",
        "vectorbt>=0.25.0",
        "quantstats>=0.0.62",
        "empyrical-reloaded>=0.5.7",
        "sqlalchemy[asyncio]>=2.0.0",
        "asyncpg>=0.29.0",
        "torch>=2.0.0",
        "joblib>=1.3.0",
        "scikit-learn>=1.3.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "PyYAML>=6.0",
        "python-dateutil>=2.8.0",
        "colorlog>=6.7.0"
    ]

setup(
    name="kallos-portfolios",
    version="1.0.0",
    author="Jose Marquez Jaramillo",
    author_email="josemarquezjaramillo@gmail.com",
    description="Cryptocurrency Portfolio Construction & Backtesting Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/josemarquezjaramillo/kallos-models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0"
        ]
    },
    include_package_data=True,
    package_data={
        "kallos_portfolios": [
            "config/*.yaml",
            "config/*.yml"
        ]
    },
    entry_points={
        "console_scripts": [
            "kallos-portfolios=kallos_portfolios.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/josemarquezjaramillo/kallos-models/issues",
        "Source": "https://github.com/josemarquezjaramillo/kallos-models",
        "Documentation": "https://github.com/josemarquezjaramillo/kallos-models/blob/main/README.md",
    },
    keywords=[
        "cryptocurrency",
        "portfolio optimization",
        "machine learning",
        "backtesting",
        "finance",
        "investment",
        "gru",
        "neural networks",
        "quantitative finance",
        "risk management",
        "asset allocation"
    ],
    zip_safe=False,
)
