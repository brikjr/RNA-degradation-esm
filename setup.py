from setuptools import setup, find_packages

setup(
    name="rna_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "fair-esm",
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "tensorboard>=2.4.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "ipykernel>=5.1.0",
        "tqdm==4.67.1",
        "viennarna",
    ],
    author="Rishabh Pandey",
    author_email="brikjr@gmail.com",
    description="RNA Structure Prediction with Few-Shot Learning",
    python_requires=">=3.10",
)