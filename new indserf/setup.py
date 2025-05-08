from setuptools import setup, find_packages
import os

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Read README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='indserf-unsupervised',
    version='1.0.0',
    description='Unsupervised Learning Pipeline for Trading Pattern Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Indserf Team',
    author_email='contact@indserf.com',
    url='https://github.com/indserf/unsupervised-trading',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'indserf-train=main:main',
            'indserf-generate-data=utils.synthetic_data:main',
        ],
    },
    # Additional package data
    package_data={
        'indserf': [
            'config/*.json',
            'models/*.pth',
            'examples/*.py',
        ],
    },
    # Dependencies for different setups
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'pytest-cov>=2.12.1',
            'black>=21.7b0',
            'flake8>=3.9.2',
            'mypy>=0.910',
            'sphinx>=4.1.2',
            'sphinx-rtd-theme>=0.5.2',
        ],
        'gpu': [
            'torch>=1.9.0+cu111',  # CUDA 11.1 version
        ],
        'viz': [
            'plotly>=5.1.0',
            'dash>=2.0.0',
            'jupyter>=1.0.0',
        ],
    },
)

# Post-installation setup
def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data',
        'models',
        'results',
        'logs',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == '__main__':
    setup_directories()
    print("\nDirectory structure created successfully!")
    print("\nTo install the package, run:")
    print("pip install -e .")
    print("\nFor development installation with extra tools:")
    print("pip install -e .[dev]")
    print("\nFor GPU support:")
    print("pip install -e .[gpu]")
    print("\nFor visualization tools:")
    print("pip install -e .[viz]")
