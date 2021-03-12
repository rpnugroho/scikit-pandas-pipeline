import setuptools

VERSION = "0.0.1"
DESCRIPTION = "Scikit-learn pipeline helper for pandas dataframe."
LONG_DESCRIPTION = "Scikit-learn pipeline helper to return dataframe instead numpy array."

setuptools.setup(
    name="skpdspipe",
    version=VERSION,
    author="Rizky Nugroho",
    author_email="nugroho.rizkyp@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # url="https://github.com/pypa/sampleproject",
    install_requires=['numpy', 'pandas', 'sklearn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
