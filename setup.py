from setuptools import find_packages, setup

setup(
    name="styletrf",
    version="0.0",
    author="Adrien Lafage",
    author_email="adrienlafage@outlook.com",
    description="A toolbox to apply easily style transfert methods",
    packages=find_packages(),
    install_requires=["torch", "torchvision"],
)
