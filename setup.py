from setuptools import setup, find_packages

setup(
    name='rtc_vision_toolbox',
    version='1.0',
    packages=find_packages(),  # Automatically find all packages (e.g., 'core')
    install_requires=[         # Add dependencies from requirements.txt
        line.strip() for line in open('requirements.txt').readlines()
    ]
)
