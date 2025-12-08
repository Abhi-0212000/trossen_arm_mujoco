"""Setup script for editable installs."""
from setuptools import setup, find_packages

setup(
    name="trossen-arm-mujoco",
    version="0.0.0",
    packages=find_packages(include=["trossen_arm_mujoco", "trossen_arm_mujoco.*"]),
    package_data={
        "trossen_arm_mujoco": ["assets/**/*"],
    },
    include_package_data=True,
    install_requires=[
        "dm_control",
        "dm_env",
        "h5py",
        "matplotlib",
        "numpy",
        "tqdm",
        "opencv-python",
        "pyquaternion",
        "scipy",
    ],
    python_requires=">=3.10",
)
