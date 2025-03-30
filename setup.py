from setuptools import setup, find_packages

setup(
    name="semegm", 
    version="0.1.0",
    author="Valentinos Pariza",
    author_email="valentinos.pariza@utn.de",
    description="A library for semantic segmentation evaluations of vision encoders.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Change if hosted on GitHub
    packages=find_packages(),  
    install_requires=[
        "faiss-cpu",
        "joblib",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
