from io import open
import setuptools

long_description = "Constructs a Knowledge Base from given corpus"
with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="kbuilder",
    version="0.0.1",
    author="Soh Wee Tee",
    author_email="weeteesoh345@gmail.com",
    description="Knowledge Base Builder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="knowledge base text mining relation extraction",
    licence="Apache",
    url="https://github.com/plkmo/Knowledge-Base-Builder",
    packages=setuptools.find_packages(exclude=["data"\
                                               "results",\
                                               "kbuilder_django"
                                               ]),
    install_requires=required,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
