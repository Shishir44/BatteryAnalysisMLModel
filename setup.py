from setuptools import setup, find_packages

setup(
    name="battery_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'xgboost>=1.5.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.0',
    ],
    author="Sugam Karki",
    description="Battery Analysis ML Project",
    python_requires=">=3.7",
)