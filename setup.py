from setuptools import setup, find_packages

setup(
    name="goai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pygame',
        'colorama',
        'numpy',
        'torch',
        'pytest',
    ],
    test_suite='ai.tests',
) 