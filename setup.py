from setuptools import setup, find_packages
setup(
    name='3d-logs-seg',
    version='0.1.0',
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    python_requires='>=3.10',
)
