from setuptools import setup, find_packages

setup(
    name="fastmc",
    version="1.0.0",
    description="A High-Performance Surrogate Tool for Fast Monte Carlo Simulations",
    author="FastMC Team",
    packages=find_packages(),
    py_modules=["fastmc", "Simulator", "experiment", "parser"],
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "tensorflow>=2.10.0",
        "matplotlib>=3.6.0",
        "joblib>=1.2.0",
        "fpdf>=1.7.2",
    ],
    entry_points={
        "console_scripts": [
            "fastmc=fastmc:main",
        ],
    },
    python_requires=">=3.8",
)
