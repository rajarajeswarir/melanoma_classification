# This is the setup file for the module melanoma_classification.

from setuptools import find_packages
from setuptools import setup

setup(name='melanoma_classification',
      version="0.0.1",
      description="Melanoma Classification",
      license="MIT",
      author="Awesome Le Wagon Team: SRS",
      author_email="contact@lewagon.org",
      #url="https://github.com/lewagon/taxi-fare",
      #install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
