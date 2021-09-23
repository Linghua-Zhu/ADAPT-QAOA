#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from setuptools import setup, find_packages

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(name='of-adapt-vqe',
      version=0,
      description='Run ADAPT-VQE with Psi4 and Openfermion',
      url='https://github.com/hrgrimsl/OF_VQE.git',
      author='Harper R. Grimsley, Nick Mayhall',
      author_email='hrgrimsl@vt.edu',
      license='Apache 2',
      install_requires=["requirements"],
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      include_package_data=True,
    )

