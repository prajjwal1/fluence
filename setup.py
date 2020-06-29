import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Fluence",
    version="0.1.5",
    author="Prajjwal Bhargava",
    author_email="prajjwalin@pm.me",
    description=(
        "Pytorch based deep learning library focussed on providing computationally"
        " efficient low resource methods and algorithms"
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="attention pytorch transformers",
    license="Apache",
    url="https://github.com/prajjwal1/fluence",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "dataclasses;python_version<'3.7'",
        "tqdm >= 4.27",
        "transformers",
        "higher",
        "pandas",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
