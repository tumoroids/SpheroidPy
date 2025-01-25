from setuptools import setup, find_packages

setup(
    name="SpheroidPy",
    version="0.1.0",
    description="Python Package for the Analysis of Spheroid Imaging Data",
    long_description=open("README.md").read(),# Lange Beschreibung (z. B. aus README.md)
    long_description_content_type="text/markdown",  # Format der langen Beschreibung

    author="Cedric Heuermann",                       # Autor
    author_email="cedric.heuermann@alumni.uni-heidelberg.de",   # E-Mail-Adresse
    url="https://github.com/CedHe/SpheroidPy",   # URL des Projekts


    packages=find_packages(),

    classifiers=[                             # Metadaten für PyPI
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    python_requires=">=3.6",


    package_data={
        'SpheroidPy': ['weights/*.pth'],
    },
    include_package_data=True,
    # ... other setup configurations

) 