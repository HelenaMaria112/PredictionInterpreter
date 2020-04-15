import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ClassificationPredictionInterpreter-pkg-HelenaMaria",
    version="0.0.1",
    author="HelenaMaria",
    author_email="anna.schmitt.anna@gmail.com",
    description="univariate model's Interpretation Techniques - also for classificating features and predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # maybe not applicable...
)