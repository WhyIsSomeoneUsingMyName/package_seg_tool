from setuptools import setup, find_packages

setup(
    name="package_seg",
    version="0.1.0",
    description="包裹分割与跟踪工具包",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'package_seg': ['configs/*.yaml', 'checkpoints/*.pt'],
    },
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python",
        "numpy",
        "tqdm",
        "pyyaml",
        "torch>=1.8.0",
        "Pillow",
        "lap>=0.5.12",
    ],
    python_requires=">=3.8",
)